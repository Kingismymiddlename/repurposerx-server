import os, re, json, httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

@app.get("/health")
def health():
    return {"status": "ok", "tool": "RepurposeRx"}

@app.get("/opentargets-drugs")
async def opentargets_drugs(disease: str):
    """Get known drugs and targets for a disease from OpenTargets"""
    result = {"disease_name": disease, "known_drugs": [], "targets": []}
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            # Search for disease EFO
            search = await client.post(
                "https://api.platform.opentargets.org/api/v4/graphql",
                json={"query": '{ search(queryString: "' + disease + '", entityNames: ["disease"]) { hits { id name } } }'},
                headers={"Content-Type": "application/json"}
            )
            if search.status_code == 200:
                hits = search.json().get("data", {}).get("search", {}).get("hits", [])
                if hits:
                    efo_id = hits[0]["id"]
                    result["disease_name"] = hits[0]["name"]
                    # Get associated drugs via known drugs query
                    drug_query = """
                    query($efoId: String!) {
                      disease(efoId: $efoId) {
                        knownDrugs(size: 20) {
                          rows {
                            drug { id name drugType maximumClinicalTrialPhase}
                            target { approvedSymbol approvedName }
                            phase
                            status
                            mechanismOfAction
                          }
                        }
                        associatedTargets(page: {index: 0, size: 10}) {
                          rows {
                            score
                            target { approvedSymbol approvedName biotype }
                          }
                        }
                      }
                    }
                    """
                    gql = await client.post(
                        "https://api.platform.opentargets.org/api/v4/graphql",
                        json={"query": drug_query, "variables": {"efoId": efo_id}},
                        headers={"Content-Type": "application/json"}
                    )
                    if gql.status_code == 200:
                        data = gql.json().get("data", {}).get("disease", {})
                        for row in (data.get("knownDrugs", {}).get("rows", []) or []):
                            d = row.get("drug", {})
                            t = row.get("target", {})
                            result["known_drugs"].append({
                                "drug": d.get("name", ""),
                                "type": d.get("drugType", ""),
                                "phase": row.get("phase", ""),
                                "status": row.get("status", ""),
                                "mechanism": row.get("mechanismOfAction", ""),
                                "target": t.get("approvedSymbol", ""),
                                "target_name": t.get("approvedName", "")
                            })
                        for row in (data.get("associatedTargets", {}).get("rows", []) or []):
                            t = row.get("target", {})
                            result["targets"].append({
                                "symbol": t.get("approvedSymbol", ""),
                                "name": t.get("approvedName", ""),
                                "score": round(row.get("score", 0), 3)
                            })
    except Exception as e:
        result["error"] = str(e)
    return result

@app.get("/chembl-approved")
async def chembl_approved(mechanism: str):
    """Search ChEMBL for approved drugs with a given mechanism"""
    results = []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://www.ebi.ac.uk/chembl/api/data/molecule",
                params={"max_phase": 4, "molecule_type": "Small molecule", "pref_name__icontains": mechanism[:30], "format": "json", "limit": 8}
            )
            if resp.status_code == 200:
                for m in resp.json().get("molecules", []):
                    p = m.get("molecule_properties", {}) or {}
                    results.append({
                        "name": m.get("pref_name", ""),
                        "chembl_id": m.get("molecule_chembl_id", ""),
                        "max_phase": m.get("max_phase", ""),
                        "indication": m.get("indication_class", "") or "",
                        "mw": p.get("full_mwt", ""),
                        "alogp": p.get("alogp", "")
                    })
    except Exception as e:
        results = [{"error": str(e)}]
    return results

@app.get("/pubmed-repurpose")
async def pubmed_repurpose(disease: str):
    results = []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            search = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pubmed", "term": f"{disease} drug repurposing repositioning 2020:2025[dp]", "retmax": 8, "retmode": "json", "sort": "relevance"}
            )
            ids = search.json().get("esearchresult", {}).get("idlist", [])
            if ids:
                fetch = await client.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                    params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract"}
                )
                xml = fetch.text
                titles = re.findall(r'<ArticleTitle>(.*?)</ArticleTitle>', xml, re.DOTALL)
                abstracts = re.findall(r'<AbstractText[^>]*>(.*?)</AbstractText>', xml, re.DOTALL)
                years = re.findall(r'<PubDate>.*?<Year>(\d{4})</Year>', xml, re.DOTALL)
                for i, t in enumerate(titles[:8]):
                    results.append({
                        "title": re.sub(r'<[^>]+>', '', t).strip(),
                        "abstract": re.sub(r'<[^>]+>', '', abstracts[i] if i < len(abstracts) else "").strip()[:450],
                        "year": years[i] if i < len(years) else ""
                    })
    except Exception as e:
        results = [{"error": str(e)}]
    return results

class RepurposeRequest(BaseModel):
    disease: str
    known_drugs: list = []
    targets: list = []
    papers: list = []
    priority: str = "efficacy"

@app.post("/analyze")
async def analyze(req: RepurposeRequest):
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured"}

    drugs_str = json.dumps(req.known_drugs[:15], indent=2)
    targets_str = json.dumps(req.targets[:10], indent=2)
    papers_str = json.dumps([{"title": p.get("title",""), "abstract": p.get("abstract","")} for p in req.papers[:6]], indent=2)

    prompt = f"""You are an expert pharmacologist and drug repurposing scientist. Analyze this disease landscape and identify approved drugs from other therapeutic areas that could be repurposed.

Disease/Condition: {req.disease}
Repurposing Priority: {req.priority}

Currently Known Drugs for This Disease (from OpenTargets):
{drugs_str}

Key Disease Targets:
{targets_str}

Recent Repurposing Literature:
{papers_str}

Your task: Identify approved drugs (already on market for OTHER indications) that have mechanistic rationale for efficacy in {req.disease}.

Respond ONLY with a raw JSON object. No markdown. No code fences.

{{
  "repurposing_candidates": [
    {{
      "drug_name": "approved drug name",
      "current_indication": "what it is currently approved for",
      "repurposing_rationale": "2-sentence mechanistic explanation of why it could work for this disease",
      "target_overlap": "which shared target/pathway makes this relevant",
      "evidence_strength": "Strong/Moderate/Weak/Theoretical",
      "development_stage": "Preclinical/Phase 1/Phase 2/Phase 3/Approved (off-label)/Literature only",
      "safety_advantage": "key safety or formulation advantage vs de novo drug",
      "timeline_to_clinic": "estimated time advantage vs new drug (e.g. 3-5 years vs 12-15 years)",
      "drug_class": "e.g. mTOR inhibitor, JAK inhibitor, monoclonal antibody",
      "repurposing_score": integer 0-100
    }}
  ],
  "disease_mechanism_summary": "2-sentence summary of the key mechanisms driving this disease that create repurposing opportunities",
  "best_opportunity": "Name of the single most promising repurposing candidate and one sentence why",
  "key_pathways_for_repurposing": ["pathway1", "pathway2", "pathway3"],
  "challenges": "2-sentence description of key regulatory and clinical challenges for repurposing in this disease",
  "time_cost_advantage": "1-2 sentence summary of the overall time/cost advantage of repurposing vs de novo for this disease",
  "confidence": integer 0-100
}}"""

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            GROQ_BASE,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": [
                {"role": "system", "content": "You are a JSON API for drug repurposing analysis. Output only raw JSON. No markdown."},
                {"role": "user", "content": prompt}
            ], "temperature": 0.2, "max_tokens": 2000, "response_format": {"type": "json_object"}}
        )
        data = resp.json()
        if "error" in data:
            return {"error": data["error"].get("message", "Groq error")}
        text = data.get("choices",[{}])[0].get("message",{}).get("content","").strip()
        text = re.sub(r'^```json\s*','',text); text = re.sub(r'^```\s*','',text); text = re.sub(r'\s*```$','',text).strip()
        try:
            return json.loads(text)
        except:
            m = re.search(r'\{[\s\S]*\}', text)
            if m:
                try: return json.loads(m.group())
                except: pass
        return {"error": f"Parse error: {text[:200]}"}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

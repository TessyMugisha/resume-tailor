import json
from io import BytesIO
from xml.sax.saxutils import escape

import streamlit as st
from anthropic import Anthropic
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable, ListFlowable, ListItem)

MODEL = "claude-haiku-4-5-20251001"
client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])


# ---------- Claude helpers ----------

def ask_claude(prompt, max_tokens):
    msg = client.messages.create(model=MODEL, max_tokens=max_tokens,
                                 messages=[{"role": "user", "content": prompt}])
    text = msg.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def parse_resume(resume_text):
    """Turn a pasted resume into the same structure as master_profile.yaml.
    Bullets are copied word for word. Nothing is invented."""
    prompt = f"""Convert this resume into JSON. Copy every bullet WORD FOR WORD.
Do not rewrite, improve, or invent anything. If a field is missing, use "" or [].

<resume>
{resume_text}
</resume>

Respond with only JSON in this exact shape:
{{"contact": {{"name": "", "email": "", "phone": "", "location": "", "linkedin": "", "github": "", "website": ""}},
 "summaries": {{"default": "summary text from the resume, or empty"}},
 "experience": [{{"title": "", "company": "", "dates": "", "location": "", "bullets": [{{"text": ""}}]}}],
 "projects": [{{"name": "", "role": "", "dates": "", "bullets": [{{"text": ""}}]}}],
 "education": [{{"degree": "", "school": "", "graduation": "", "location": ""}}],
 "skills": {{"all": ["skill"]}},
 "certifications": [{{"name": "", "issuer": "", "date": ""}}],
 "affiliations": [{{"name": ""}}],
 "leadership": [{{"text": ""}}]}}
"""
    profile = ask_claude(prompt, 4000)
    for key, default in [("summaries", {"default": ""}), ("experience", []), ("projects", []),
                         ("education", []), ("skills", {"all": []}), ("certifications", []),
                         ("affiliations", []), ("leadership", [])]:
        profile.setdefault(key, default)
    if not profile["summaries"]:
        profile["summaries"] = {"default": ""}
    return profile


def build_catalog(profile):
    catalog, lines = {}, []
    for i, job in enumerate(profile["experience"]):
        lines.append(f"\n[JOB] {job['title']} at {job['company']} ({job['dates']})")
        for j, b in enumerate(job["bullets"]):
            catalog[f"exp{i}_{j}"] = b["text"]
            lines.append(f"exp{i}_{j}: {b['text']}")
    for i, proj in enumerate(profile["projects"]):
        lines.append(f"\n[PROJECT] {proj['name']} ({proj['dates']})")
        for j, b in enumerate(proj["bullets"]):
            catalog[f"proj{i}_{j}"] = b["text"]
            lines.append(f"proj{i}_{j}: {b['text']}")
    return catalog, "\n".join(lines)


def tailor(profile, catalog_text, job_posting):
    certs = "\n".join(f"cert{i}: {c['name']} ({c['issuer']})" for i, c in enumerate(profile["certifications"]))
    affs = "\n".join(f"aff{i}: {a['name']}" for i, a in enumerate(profile["affiliations"]))
    leads = "\n".join(f"lead{i}: {l['text']}" for i, l in enumerate(profile["leadership"]))
    all_skills = [s for g in profile["skills"].values() for s in g]
    prompt = f"""Pick the best resume content for this job posting.

<job_posting>
{job_posting}
</job_posting>

<summaries>
{json.dumps(profile["summaries"], indent=2)}
</summaries>

<bullets>
{catalog_text}
</bullets>

<certifications>
{certs}
</certifications>

<affiliations>
{affs}
</affiliations>

<leadership>
{leads}
</leadership>

Rules:
- Pick 12 to 15 bullet IDs total, or all of them if there are fewer.
- Give each chosen job at least 2 bullets when possible.
- Pick the best summary key.
- Pick up to 10 skills from this list only: {json.dumps(all_skills)}
- Pick up to 3 certification IDs, up to 5 affiliation IDs, and up to 2 leadership IDs that fit this job.

Respond with only JSON in this exact shape, and nothing else:
{{"summary_key": "default", "bullet_ids": ["exp0_1"], "cert_ids": ["cert0"], "affiliation_ids": ["aff0"], "leadership_ids": ["lead0"], "skills_to_highlight": ["Python"], "why": "one sentence on your choices"}}
"""
    return ask_claude(prompt, 1000)


# ---------- PDF ----------

styles = {
    "name": ParagraphStyle("name", fontName="Times-Bold", fontSize=20, leading=24, alignment=TA_CENTER),
    "contact": ParagraphStyle("contact", fontName="Times-Roman", fontSize=10, leading=13, alignment=TA_CENTER),
    "section": ParagraphStyle("section", fontName="Times-Bold", fontSize=11.5, leading=14, spaceBefore=8),
    "body": ParagraphStyle("body", fontName="Times-Roman", fontSize=10, leading=12.5),
    "left": ParagraphStyle("left", fontName="Times-Bold", fontSize=10.5, leading=13),
    "right": ParagraphStyle("right", fontName="Times-Roman", fontSize=10, leading=13, alignment=TA_RIGHT),
    "sub": ParagraphStyle("sub", fontName="Times-Italic", fontSize=10, leading=12),
}


def link(url):
    label = url.replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")
    href = url if url.startswith("http") else "https://" + url
    return f'<a href="{escape(href)}"><u>{escape(label)}</u></a>'


def section(story, title):
    story.append(Paragraph(title.upper(), styles["section"]))
    story.append(HRFlowable(width="100%", thickness=0.7, color=colors.black, spaceBefore=1, spaceAfter=4))


def row(left, right):
    t = Table([[Paragraph(left, styles["left"]), Paragraph(right, styles["right"])]],
              colWidths=[5.2 * inch, 2.3 * inch])
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def bullet_list(items):
    return ListFlowable([ListItem(Paragraph(escape(b), styles["body"])) for b in items],
                        bulletType="bullet", start="•", leftIndent=12, bulletFontSize=8)


def pick(prefix, items, ids):
    chosen = []
    for x in ids:
        num = x[len(prefix):]
        if x.startswith(prefix) and num.isdigit() and int(num) < len(items):
            chosen.append(items[int(num)])
    return chosen


def build_pdf(profile, catalog, result):
    picked = {bid: catalog[bid] for bid in result["bullet_ids"] if bid in catalog}
    all_skills = [s for g in profile["skills"].values() for s in g]
    skills = [s for s in result.get("skills_to_highlight", []) if s in all_skills]
    c = profile["contact"]
    story = [Paragraph(escape(c.get("name", "")), styles["name"])]

    line1 = " | ".join(escape(c[k]) for k in ["phone", "email", "location"] if c.get(k))
    line2 = " | ".join(link(c[k]) for k in ["linkedin", "github", "website"] if c.get(k))
    for line in [line1, line2]:
        if line:
            story.append(Paragraph(line, styles["contact"]))

    summary = profile["summaries"].get(result.get("summary_key"), "") or next(iter(profile["summaries"].values()), "")
    if summary:
        section(story, "Summary")
        story.append(Paragraph(escape(" ".join(summary.split())), styles["body"]))

    if profile["education"]:
        section(story, "Education")
        for e in profile["education"]:
            story.append(row(escape(e.get("degree", "")), escape(str(e.get("graduation", "")))))
            sub = ", ".join(x for x in [e.get("school", ""), e.get("location", "")] if x)
            story.append(Paragraph(escape(sub), styles["sub"]))
            story.append(Spacer(1, 3))

    jobs = [(i, job) for i, job in enumerate(profile["experience"])]
    if any(f"exp{i}_{j}" in picked for i, job in jobs for j in range(len(job["bullets"]))):
        section(story, "Experience")
        for i, job in jobs:
            bullets = [picked[f"exp{i}_{j}"] for j in range(len(job["bullets"])) if f"exp{i}_{j}" in picked]
            if bullets:
                story.append(row(escape(job["title"]), escape(job["dates"])))
                sub = ", ".join(x for x in [job.get("company", ""), job.get("location", "")] if x)
                story.append(Paragraph(escape(sub), styles["sub"]))
                story.append(bullet_list(bullets))
                story.append(Spacer(1, 4))

    projs = [(i, p) for i, p in enumerate(profile["projects"])]
    if any(f"proj{i}_{j}" in picked for i, p in projs for j in range(len(p["bullets"]))):
        section(story, "Projects")
        for i, proj in projs:
            bullets = [picked[f"proj{i}_{j}"] for j in range(len(proj["bullets"])) if f"proj{i}_{j}" in picked]
            if bullets:
                role = f' <font name="Times-Italic">| {escape(proj["role"])}</font>' if proj.get("role") else ""
                story.append(row(f'{escape(proj["name"])}{role}', escape(proj["dates"])))
                story.append(bullet_list(bullets))
                story.append(Spacer(1, 4))

    leads = pick("lead", profile["leadership"], result.get("leadership_ids", []))
    if leads:
        section(story, "Leadership")
        story.append(bullet_list([l["text"] for l in leads]))

    if skills:
        section(story, "Skills")
        story.append(Paragraph(escape(", ".join(skills)), styles["body"]))

    certs = pick("cert", profile["certifications"], result.get("cert_ids", []))
    if certs:
        section(story, "Certifications")
        story.append(Paragraph(escape("; ".join(f'{x["name"]}, {x["issuer"]} ({x["date"]})' for x in certs)), styles["body"]))

    affs = pick("aff", profile["affiliations"], result.get("affiliation_ids", []))
    if affs:
        section(story, "Affiliations")
        story.append(Paragraph(escape(", ".join(a["name"] for a in affs)), styles["body"]))

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=0.5 * inch, rightMargin=0.5 * inch,
                            topMargin=0.45 * inch, bottomMargin=0.45 * inch,
                            title=f'{c.get("name", "")} Resume', author=c.get("name", ""))
    doc.build(story)
    return buffer.getvalue(), doc.page


# ---------- Page ----------

st.set_page_config(page_title="Resume Tailor", page_icon="📄")
st.title("Resume Tailor")
st.write("Paste your full resume and a job posting. Get a tailored PDF back. "
         "It only uses bullets from your own resume. It never makes things up.")
st.caption("V1 test. Your text is sent to Claude to process and is not saved.")

def read_upload(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(file).pages)
    if name.endswith(".docx"):
        from docx import Document
        return "\n".join(p.text for p in Document(file).paragraphs)
    return file.read().decode("utf-8", errors="ignore")


st.subheader("1. Your resume")
st.caption("Upload your resume, or paste it. Include everything, even old bullets. More to pick from means a better fit.")
upload = st.file_uploader("Upload a PDF, Word (.docx), or text file", type=["pdf", "docx", "txt"])
pasted = st.text_area("Or paste it here", height=220)

resume_text = ""
if upload is not None:
    try:
        resume_text = read_upload(upload)
        if len(resume_text.strip()) < 50:
            st.warning("I couldn't read much text from that file. It may be a scanned image. Try pasting instead.")
        else:
            st.success(f"Read {upload.name}.")
    except Exception:
        st.warning("I couldn't read that file. Try pasting your resume instead.")
if not resume_text.strip():
    resume_text = pasted

st.subheader("2. The job posting")
job_posting = st.text_area("Paste the full job posting", height=220)

if st.button("Tailor my resume", type="primary"):
    if not resume_text.strip() or not job_posting.strip():
        st.warning("Please paste both your resume and the job posting.")
    else:
        try:
            with st.spinner("Reading your resume..."):
                profile = parse_resume(resume_text)
                catalog, catalog_text = build_catalog(profile)
            with st.spinner("Picking the best fit for this job..."):
                result = tailor(profile, catalog_text, job_posting)
                pdf_bytes, pages = build_pdf(profile, catalog, result)
            st.success(f"Done. Picked {len(result['bullet_ids'])} bullets. {pages} page(s).")
            st.write(f"**Why these choices:** {result.get('why', '')}")
            name = profile["contact"].get("name", "resume").replace(" ", "_") or "resume"
            st.download_button("Download PDF", pdf_bytes, file_name=f"{name}_tailored.pdf",
                               mime="application/pdf")
        except Exception as e:
            st.error("Something went wrong. Try again, or send Tessy a screenshot of this message.")
            st.caption(str(e))

st.divider()
st.write("**Tester feedback:** What confused you? What would you want next? Tell Tessy directly.")

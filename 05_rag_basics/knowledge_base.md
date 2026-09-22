# The Institute for Adaptive Systems (IAS) — Internal Handbook

*This document is fictional. It exists so that the models in this workshop
cannot possibly have memorised it which is exactly what makes it a fair test
of retrieval.*

## About the institute

The Institute for Adaptive Systems (IAS) was founded in 1998 in Konstanz. It
employs 214 people across four departments and is directed by Prof. Miriam
Falkenrath, who took office in 2021. The institute's annual budget for 2025 is
18.4 million euros, of which 62% comes from competitive third-party grants.

## Departments

**Department A — Collective Behaviour.** Led by Dr. Yusuf Ahmadi. Studies
flocking, shoaling and swarm decision-making in birds and fish. Operates the
Tracking Hall, a 400 square metre motion-capture facility.

**Department B — Computational Cognition.** Led by Dr. Lena Vogt. Builds
computational models of decision making under uncertainty. Maintains the
institute's GPU cluster, named HELIOS.

**Department C — Environmental Sensing.** Led by Dr. Tomás Reinhardt. Runs
long-term field stations on Lake Constance and in the Bavarian Forest.

**Department D — Methods and Statistics.** Led by Dr. Priya Nandakumar. Provides
statistical consulting to the other three departments and teaches the
institute's methods curriculum.

## The HELIOS compute cluster

HELIOS has 96 nodes. Each node has 4 NVIDIA H100 GPUs and 512 GB of RAM.
Jobs are submitted through SLURM. The default wall-time limit is 24 hours;
extensions up to 72 hours require approval from the department head.

Storage is split into three tiers:

- `/home` — 100 GB per user, backed up nightly, never purged.
- `/scratch` — 20 TB shared, NOT backed up, files older than 30 days are
  deleted automatically every Sunday at 03:00.
- `/archive` — tape storage, request via the helpdesk ticket system, retrieval
  takes up to 48 hours.

Requesting a HELIOS account requires a signed form countersigned by your
supervisor. Accounts are activated within three working days. Accounts of people
who have left the institute are disabled after 60 days and deleted after 180.

## Data management policy (revised March 2024)

All research data must be registered in the institute's data catalogue within
14 days of collection. Every dataset needs a DOI before the associated paper is
submitted. Raw data must be retained for 10 years after publication.

Personal data is governed by the DPA-2024 internal standard: it must be
pseudonymised at collection, stored only on `/home` or `/archive` (never
`/scratch`), and may not leave EU-hosted infrastructure. Any transfer to a
non-EU collaborator requires prior approval from the data protection officer,
Dr. Anke Brenner.

Violations of DPA-2024 must be reported within 72 hours.

## Publication policy

IAS mandates green open access: an accepted manuscript must be deposited in the
institutional repository within 30 days of acceptance. The institute covers
article processing charges up to 2,500 euros per paper, with a maximum of two
papers per first author per calendar year.

Authorship follows the CRediT taxonomy. Disputes are mediated by the Ombudsperson,
currently Dr. Priya Nandakumar. Preprints are encouraged and should be posted
before or at submission.

## Travel and conference funding

PhD students receive 1,800 euros per year for conference travel; postdocs
receive 2,400 euros. Applications go through the travel portal at least 21 days
before departure. Flights under 700 km must be booked as rail travel instead,
under the institute's 2023 sustainability policy.

Reimbursement claims must be submitted within 8 weeks of return, with original
receipts. Per diem rates follow the federal schedule.

## The PhD programme

The IAS PhD programme runs for 4 years. Students form a Thesis Advisory
Committee (TAC) of three members by the end of month 6. The TAC meets at least
once every 12 months. A written progress report is due two weeks before each
meeting.

The mandatory curriculum is 12 ECTS: 6 from methods courses, 3 from a
transferable-skills course, and 3 from teaching or outreach. The Good Scientific
Practice course must be completed in the first year.

Submission requires two internal reviewers and one external reviewer. The
defence is public and lasts 90 minutes: 30 minutes presentation, 60 minutes
questions.

## Ethics approval

Any study involving human participants requires approval from the IAS Ethics
Committee, which meets on the first Tuesday of each month. Applications close
14 days before the meeting. Animal work requires a separate permit from the
regional authority, which typically takes 4 to 6 months.

Retrospective approval is never granted. Studies started without approval cannot
be published under the IAS affiliation.

# Ambiguity Log

Tracks source-content issues found while sourcing `evaluation_questions.csv`
that were NOT used as evidence for any question. Kept separate from the
ground-truth CSV so ambiguous content never silently becomes "verified"
evidence -- this file is the record of what was deliberately excluded, and
why, for methodology write-up purposes.

---

## 1. Broken FN/NRI eligibility fragment

- **Source URL:** https://www.gitam.edu/international-admissions
- **Category context:** Admissions/eligibility (relevant to ADM-06)
- **Raw excerpt (from cleaned corpus text):**
  > "A student pursuing their qualifying examination in India. Citizens,
  > whose either or both parents having NRI status."
- **Issue:** Grammatically incomplete/garbled. Appears to be a fragment of
  a 4th eligibility category (likely describing children of NRI parents
  studying in India) but the sentence structure is broken in the source
  page's own text -- confirmed present in the full cleaned page text
  (verified via `verify_pilot_candidate.py`), so this is a source/cleaning
  artifact, not something introduced by chunking.
- **Decision:** Excluded from all evidence. Not used for ADM-06 (which
  uses only the separate, clean "Foreign National (FN)" sentence on the
  same page) or any other question.

---

## 2. Garbled refund-policy table

- **Source URL:** https://www.gitam.edu/gimsr/admissions/under-graduate
- **Category context:** Admissions/eligibility (relevant to ADM-07)
- **Raw excerpt (from cleaned corpus text):**
  > "Time of withdrawal / Amount / The amount of fees to be deducted by
  > GIMSR on re-allocation of seat to the candidates by MCC. / Rs.20,000/- /
  > The Amount of fees to be reimbursed in case of candidate resigns after
  > counseling period. / No refund of fee / Time for refund of fee. (after
  > deducting Rs. 20,000/- towards admission processing charges). / 5
  > working days"
- **Issue:** This is an HTML table converted to flat text during cleaning;
  row/column structure is not reliably recoverable from the flattened text
  alone. A real fact likely exists here (a ₹20,000 processing charge is
  deducted on re-allocation; refund processing takes 5 working days if
  applicable) but confidence in correctly attributing which amount belongs
  to which row is not high enough for evaluation-question use.
- **Decision:** Excluded from all evidence. Not used for ADM-07 (which
  uses only the separate, clean NEET-UG/MCC counseling sentence at the top
  of the same page).

---

## 3. Internal contradictions on the scholarships policy page

- **Source URL:** https://www.gitam.edu/fee-scholarship/student-scholarships
- **Category context:** Fees/scholarships (relevant to FEE-03; explicitly avoided for FEE-03)
- **Issue:** This page appears to concatenate multiple policy-document
  versions (likely different academic-year revisions merged during
  scraping/cleaning, or the source page itself mixes historical and
  current policy text) into one page, producing genuinely conflicting
  figures in different sections of the SAME page:
  1. Need-based scholarship "Tier A" income threshold: stated as
     **Rs. 16.00 LPA** in one section, but **Rs. 12.00 LPA** in two other
     sections (a numbered-list restatement and the FAQ section).
  2. Need-based document submission deadline: stated as both
     **"July 31, 2026"** and **"31 August 2025"** in different sections.
  3. Employee Children Scholarship slabs: stated as **"60%, 40%, and 20%"**
     in one place, **"60%, 40%, 25%, and 15%"** in another.
- **Decision:** None of these three specific figures used as evidence for
  any question. Only the "Combined Scholarship Cap" figure (75%, two named
  exceptions) was used for FEE-03, specifically because it was found
  consistently restated across two separate sections, unlike the figures
  above.

---

## 4. Unclear campus attribution on the general fee-structure page

- **Source URL:** https://www.gitam.edu/fee-scholarship/fee-structure
- **Category context:** Fees/scholarships (relevant to FEE-04; UG/PG figures avoided)
- **Issue:** The page opens with "Select a campus to view fee details",
  indicating a campus-selector UI. Since the crawler captures
  server-rendered HTML (not post-interaction JS states), the UG/PG
  programme-specific fees shown (e.g. B.Tech CSE at Rs. 2,04,200) cannot
  be confidently attributed to a specific campus -- and campus-dependent
  fee variation is already confirmed elsewhere in the corpus (see FEE-02,
  where the same MBA programme costs different amounts at different
  campuses).
- **Decision:** UG/PG programme-specific fees on this page excluded from
  all evidence. Only the PhD fee-structure table was used (FEE-04),
  because that exact figure was independently corroborated as identical
  across multiple discipline blocks on this same page AND on separate
  campus-specific PhD course pages, giving confidence it is genuinely
  campus-invariant rather than an artifact of unclear attribution.

---

---

## 5. Systemic missing eligibility criterion on B.Tech course pages

- **Source URLs:** at least AI&ML B.Tech (Bengaluru), and this same pattern
  was observed earlier (pre-dating this evaluation set) on Physics,
  Chemistry, and Civil Engineering B.Tech pages during pilot testing.
- **Issue:** The "Eligibility" section on B.Tech course pages consistently
  starts at item **"b."**, with no "a." present -- e.g. "Eligibility b.
  Score over 55% (110/200) in GAT. c. Score over 85% in X Standard."
  This is a systemic pattern across multiple pages/disciplines, not a
  one-off cleaning artifact -- likely a missing first bullet (probably a
  degree/qualification criterion) that isn't being captured by the
  crawler/cleaner, or is missing in the source HTML's list markup itself.
- **Decision:** No B.Tech "Eligibility" section text used as evidence for
  any question (PRG-02 deliberately avoided this section on its source
  page). This affects any FUTURE question sourced from a B.Tech course
  page's eligibility section too -- worth checking for this pattern before
  using such a section anywhere else in the evaluation set.

---

---

## 6. Identical "placement statistics" block repeated across unrelated departments

- **Source URLs (confirmed identical block, at least 5 independent pages):**
  Physics (Visakhapatnam), Physics (Hyderabad), Chemistry (Visakhapatnam),
  Psychology (Visakhapatnam), AI&DS (Hyderabad), and Applied Psychology
  (Bengaluru).
- **Category context:** Placements/careers -- this finding is the reason
  NO per-department placement-statistics table was used as ground truth
  anywhere in this category (PLC-01 through PLC-04 all deliberately use
  other content instead).
- **Issue:** Every one of these unrelated department pages displays the
  exact same numbers under a "Class of 2021 2022 2023 2024" placement
  table: No. of Recruiters 66/85/65/68, Number of Offers 246/233/250/143,
  Number of Students Placed 219/190/201/123, Highest Salary Package
  12/9.8/13.58/16.22 (LPA), Average Salary Package 6/5.97/6.44/6.7 (LPA).
  It is not plausible that Physics, Chemistry, Psychology, and AI&DS
  independently produced identical placement outcomes across four
  different years. This strongly indicates a generic/template stat block
  (likely a dynamic widget that failed to populate department-specific
  data during the static crawl) rather than genuine per-department data.
- **Decision:** No numeric figure from this specific stat-table pattern
  (the "Class of [year]... No. of Recruiters... Number of Offers..."
  block) is used as ground truth for ANY question, in any category, for
  any department. This applies going forward to Leadership/Faculty,
  Facilities, and any other category too, not just Placements -- worth
  checking for this exact pattern before sourcing evidence from any
  department/course page's stats section.

---

---

## 7. Title inconsistency for the same named individual across campus pages

- **Person:** S Arun Kumar
- **Source URLs:**
  - `bengaluru/gitam-school-of-computer-science-and-engineering` — titled
    **"Dean – GITAM School of Computer Science and Engineering"**
  - `visakhapatnam/gitam-school-of-computer-science-and-engineering`
    (seen earlier in pilot-diagnostic output) — titled **"Director"**
    instead, for the same school/person
- **Category context:** Leadership/faculty
- **Issue:** The same person is given different formal titles ("Dean" vs
  "Director") on different campus pages for what appears to be the same
  cross-campus role. Not resolvable from the corpus alone which title is
  authoritative/current.
- **Decision:** S Arun Kumar not used as the answer for any "who is the
  Dean" or "who is the Director" question in this evaluation set. LDR-01
  uses Vamsidhar Yendapalli instead (Bengaluru-specific Director title,
  no cross-page conflict found for this person).

---

## 8. Second placement-statistics template variant

- **Source URL:** `bengaluru/gitam-school-of-computer-science-and-engineering`
- **Category context:** Leadership/faculty sourcing (incidental finding;
  page was used for LDR-01, but NOT for its placement stats)
- **Issue:** This page has its own "Placements" stat block, structurally
  similar to the pattern already flagged in Ambiguity #6 (Class of [year]
  / Number of Recruiters / Number of Offers / Salary figures), but with
  DIFFERENT numbers and a 5-year span (2020-2024) rather than the 4-year
  span (2021-2024) seen on the Physics/Chemistry/Psychology pages. This
  suggests there may be more than one template variant in circulation
  across the site (e.g. one per "School", reused across that School's
  department pages), rather than a single sitewide constant block.
- **Decision:** Does not change the decision already recorded in
  Ambiguity #6 -- no placement-statistics figures from ANY such block
  (regardless of variant) are used as ground truth anywhere in this
  evaluation set. Recorded here only to note that the pattern is more
  structurally varied than initially observed, which may be relevant if
  this is written up in the methodology section.

---

---

## 9. Missing contact details in anti-ragging section (Bengaluru)

- **Source URL:** `bengaluru/campus-life/residential-life`
- **Category context:** Not used for any Facilities question; flagged for
  awareness heading into Policies/Student Support, where anti-ragging
  content is likely to be sourced.
- **Issue:** The "National Anti Ragging Helpline" and "UGC Monitoring
  Agency Centre for Youth C4Y" headings appear on this page, but the
  actual phone number/email/website text that follows them is MISSING
  from the cleaned extraction -- likely icon-linked contact details
  (phone/email/web icons with no adjacent text) that didn't survive
  HTML-to-text conversion. By contrast, the equivalent section on the
  Visakhapatnam residential-life page DOES have the full contact details
  extracted correctly (helpline "1800 180 5522", email, website).
- **Decision:** If an anti-ragging helpline question is sourced later,
  use the Visakhapatnam page (where the contact details are confirmed
  present), not the Bengaluru page (where they are missing).

---

---

## 10. Contradictory committee rosters on the same page

- **Source URL:** `academics/evaluation/grievance-redressal`
- **Category context:** Policies/student support -- page NOT used for any question.
- **Issue:** The page contains what appear to be two different Grievance
  Redressal Cell (GRC) committee rosters, with no header distinguishing
  them. The first roster names **Prof. K. Nagendra Prasad** as
  Chairperson; the second names a different person, **Prof. Challa Murali
  Mohan**, also as Chairperson. Additionally, one person (Dr. S. Sushma
  Raj) is titled "Associate Professor" in the first roster and "Director"
  of the same school in the second. No way to determine from the
  extracted text which roster is current/authoritative -- likely two
  versions of the committee (e.g. old and renewed) concatenated during
  scraping/cleaning, similar in nature to Ambiguity #3.
- **Decision:** This page is not used as ground truth for any question in
  this evaluation set (no "who chairs the GRC" question was created).

---

## 11. Policy document referenced but not present in corpus

- **Source URL:** `gimsr/committees/antiragging-policy`
- **Category context:** Policies/student support -- page NOT used for any question.
- **Issue:** Despite the URL and page length (1,473 chars) suggesting
  substantive content, the cleaned text is entirely GIMSR's site
  navigation menu (department lists, login links, etc.). The actual
  policy content is behind a "Guidelines -- Click here for document" link
  pointing to what is presumably a PDF, which was never captured as text
  in this corpus (consistent with the earlier finding that PDF-linked
  content is generally not represented -- see the admissions-eligibility
  ambiguity notes on PDF-based eligibility criteria documents found during
  early corpus searching).
- **Decision:** This page is not used as ground truth for any question.
  GIMSR-specific anti-ragging policy content is simply not available as
  text in this corpus.

---

## 12. Post-freeze source conflict discovered during generation smoke testing (LDR-03)

- **Question:** LDR-03 -- "Who is the Director of GITAM School of Science at the Visakhapatnam campus?"
- **Frozen expected answer (unchanged, not overwritten):** Chandu Kavitha,
  sourced from `visakhapatnam/physics`, verified during Leadership/Faculty
  category sourcing.
- **New finding:** During Mode B/C generation smoke testing, retrieval
  surfaced a DIFFERENT page -- `about/leadership` -- which names a
  DIFFERENT person, **S Sushma Raj**, for what reads as the same role.
  Mistral generated a confident, correctly-cited answer using this second
  page's content. This is structurally the same failure pattern as
  Ambiguity #7 (S Arun Kumar titled "Dean" on one campus page and
  "Director" on another) -- two real corpus pages naming different people
  for a role with the same/similar title -- but discovered on a different
  page pair, and only surfaced because generation was actually run against
  live retrieval rather than caught during manual sourcing.
- **Status:** LDR-03 is marked as an exploratory `conflict_case`. It is
  NOT removed from the dataset and NOT re-verified/resolved here (frozen
  ground truth stays as originally verified: Chandu Kavitha). It WILL
  still be run in the full generation evaluation, but is EXCLUDED from
  primary generation-correctness metrics (factual correctness, refusal
  correctness, etc.) and reported separately, since scoring it against
  either answer as simply "right" or "wrong" would misrepresent a genuine
  corpus-level ambiguity as a generation failure.
- **Open question for future work (not resolved now):** whether
  `about/leadership` reflects a more recent university-wide directory
  update than the department-specific `visakhapatnam/physics` page, or
  whether "Director of GITAM School of Science" is themselves distinct
  from a per-discipline departmental leadership role. Not determinable
  from the corpus alone.

---

*(This file will be appended to as further categories are sourced.)*

# User Profile

Background, goals, and ongoing context for the user.

---

## Background
- Researcher working on multimodal AI coherence evaluation
- Building a publication-quality research framework (MultiModal Coherence AI)
- Developing the cMSCI metric as a novel contribution

## Current Project State
- Phases 1-21 complete (RQ1-3, stats, figures, paper, bridge, generative, professor feedback, Gemini v2, v2 optimization, 100-sample scale-up)
- 100 samples rated by 5 raters, optimization + evaluation complete
- Papers updated (paper.md, paper_v2.md, paper_v2.tex) with 100-sample results
- v2/v3 (Gemini) re-evaluation pending on 100 samples (needs GOOGLE_API_KEY)
- Currently on `clean-for-push` branch — preparing for publication/sharing

## Goals
- Finalize and publish the research paper
- Demonstrate cMSCI as a robust, interpretable multimodal coherence metric
- Achieve statistically significant human correlation (achieved: rho=0.785, p<1e-6)

## Key Results Achieved
- cMSCI v1: rho=0.785, p<1e-6 (100 samples, 5 raters)
- LOO-CV: rho=0.749, p<1e-6 (minimal overfitting, gap=0.001)
- ICC(3,k)=0.872 (5 raters, 100 samples); ICC(3,k)=0.917 (8 raters, 30 samples)
- Beats all baselines: cosine_znorm=0.712, MSCI=0.558, RegCCA=0.495
- External benchmark AUC=0.969 on AudioCaps

---

_Update this file with evolving goals, milestones, and context._

# The Role of Generalists in Team-Level Innovation
## Overview
This paper analyzes how having more generalist inventors in a research team affects the impact of team innovation, as measured by forward citations.
## Research Problems
- Despite the importance of team-level collaboration in modern day research, most studies on generalists and specialists have been conducted on the individual level.
## Methods Overview
- Collected US patent data assigned to Global Pharmaceutical firms (with SIC code 2834) during years 2009-2015
- For each inventor on a patent, identifiend their previous patenting records by CPC subclasses to calculate the Herfindahl-Hirschman Index as a measure of portfolio generality
- Controlling for inventor-level, inventor pair-level, team-level, patent-level, and assignee-level variables, examined the association between the proportion of generalist inventors in a team and the number of citations the resulting patent received
## Results
- Proportion of generalist inventors has a negative association with innovation impact
- Generalist inventors has a more positive association when the patent is in an unfamiliar domain (unfamiliarity measured as the proportion of team members who have not previously patented in the CPC subclass the focal patent is in)
## Data Sources
[PatentsView](https://patentsview.org/): Full catalogue of USPTO patent data
[UVA Darden GCPD](https://patents.darden.virginia.edu/): Crosswalk between USPTO assignees and US Public firms
[Compustat](https://www.marketplace.spglobal.com/en/datasets/compustat-financials-(8)): Company financial data for control variables
## Tools used
Python: Data mining, cleaning, and wrangling
STATA: Negative Binomial Regression
[GROBID](https://github.com/kermitt2/grobid): Extracting text data from PDFs (WIP)
OpenAI API: Parsing relevant sections from full-text (WIP)

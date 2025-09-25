import pandas as pd
import math
import pulp
from tqdm import tqdm

class Allocator:
    def __init__(self, social_quotas=None, rural_quota=0.4, past_quota=0.9,
                 slack_penalty=1000, top_candidate_multiplier=5):
        self.social_quotas = social_quotas or {"SC":0.15,"ST":0.075,"OBC":0.27,"EWS":0.10}
        self.rural_quota = rural_quota
        self.past_quota = past_quota
        self.slack_penalty = slack_penalty
        self.top_candidate_multiplier = top_candidate_multiplier

    def get_cand_value(self, candidates, c_id, col, default=0):
        val = candidates.loc[candidates['Candidate ID']==c_id, col]
        if val.empty: return default
        try: return int(val.values[0])
        except: return default

    def get_intern_capacity(self, interns, j):
        for col in ["capacity","Capacity","Vacancy","vacancy","Seats","seats"]:
            if col in interns.columns:
                try: return max(1,int(float(interns.loc[interns['Internship ID']==j, col].values[0])))
                except: continue
        return 1

    def allocate(self, candidates, interns, scores_df):
        candidate_ids = scores_df.index.tolist()
        internship_ids = scores_df.columns.tolist()
        capacity = {j: self.get_intern_capacity(interns,j) for j in internship_ids}
        allocations = []

        for j in tqdm(internship_ids, desc="Allocating Internships"):
            cap = capacity[j]
            top_candidates = scores_df[j].sort_values(ascending=False).head(cap*self.top_candidate_multiplier).index.tolist()
            if len(top_candidates)==0: continue

            prob = pulp.LpProblem(f"Internship_{j}", pulp.LpMaximize)
            x = pulp.LpVariable.dicts("assign", top_candidates, cat=pulp.LpBinary)
            slack_social = {cat:pulp.LpVariable(f"slack_social_{cat}", lowBound=0) for cat in list(self.social_quotas.keys())+["GEN"]}
            slack_rural = pulp.LpVariable("slack_rural", lowBound=0)
            slack_past  = pulp.LpVariable("slack_past", lowBound=0)

            prob += pulp.lpSum([scores_df.at[i,j]*x[i] for i in top_candidates]) - self.slack_penalty*(
                slack_rural + slack_past + pulp.lpSum([slack_social[cat] for cat in slack_social])
            )

            prob += pulp.lpSum([x[i] for i in top_candidates]) <= cap

            required_per_cat = {cat: math.ceil(self.social_quotas.get(cat,0)*cap) for cat in self.social_quotas}
            required_per_cat["GEN"] = max(0, cap - sum(required_per_cat.values()))
            for cat, req in required_per_cat.items():
                if cat == "GEN":
                    eligible = [i for i in top_candidates if all(self.get_cand_value(candidates,i,c_col)==0
                                                                 for c_col in ["Social_Category_EWS","Social_Category_OBC","Social_Category_SC","Social_Category_ST"])]
                else:
                    col_name = f"Social_Category_{cat}"
                    eligible = [i for i in top_candidates if self.get_cand_value(candidates,i,col_name)==1]
                if len(eligible)==0:
                    prob += slack_social[cat] >= req
                else:
                    prob += pulp.lpSum([x[i] for i in eligible]) + slack_social[cat] >= req

            req_rural = math.ceil(self.rural_quota*cap)
            req_urban = math.ceil((1-self.rural_quota)*cap)
            rural_cands = [i for i in top_candidates if self.get_cand_value(candidates,i,"location_category")==0]
            urban_cands = [i for i in top_candidates if self.get_cand_value(candidates,i,"location_category")==1]
            prob += pulp.lpSum([x[i] for i in rural_cands]) + slack_rural >= req_rural
            prob += pulp.lpSum([x[i] for i in urban_cands]) + slack_rural >= req_urban

            req_new = math.ceil(self.past_quota*cap)
            new_cands = [i for i in top_candidates if self.get_cand_value(candidates,i,"Past Participation")==0]
            prob += pulp.lpSum([x[i] for i in new_cands]) + slack_past >= req_new

            prob.solve(pulp.PULP_CBC_CMD(msg=False))

            for i in top_candidates:
                if pulp.value(x[i]) is not None and round(pulp.value(x[i]))==1:
                    allocations.append({"InternshipID":j, "CandidateID":i, "Score":scores_df.at[i,j]})

        alloc_df = pd.DataFrame(allocations)
        if not alloc_df.empty:
            alloc_df["Rank"] = alloc_df.groupby("InternshipID")["Score"].rank(method="first", ascending=False).astype(int)
            alloc_df = alloc_df.sort_values(["InternshipID","Rank"]).reset_index(drop=True)
        return alloc_df

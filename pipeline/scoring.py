import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Scorer:
    def compute_skill_matrix(self, cand_tech_emb, cand_soft_emb, cand_proj_emb,
                             intern_req_emb, intern_resp_emb, intern_desc_emb):
        return (
            0.6 * cosine_similarity(cand_tech_emb, intern_req_emb) +
            0.25 * cosine_similarity(cand_proj_emb, intern_resp_emb) +
            0.15 * cosine_similarity(cand_soft_emb, intern_desc_emb)
        )

    def apply_category_adjustment(self, skill_matrix, cand_cat, intern_cat):
        cand_cat = cand_cat.str.lower().fillna("").values[:, None]
        intern_cat = intern_cat.str.lower().fillna("").values[None, :]
        cat_match_matrix = (cand_cat == intern_cat).astype(float)
        skill_matrix = skill_matrix * (0.5 + 0.6 * cat_match_matrix)
        return np.clip(skill_matrix, 0, 1)

    def apply_location_adjustment(self, skill_matrix, cand_locs, intern_locs):
        location_matrix = np.zeros_like(skill_matrix)
        cand_locs = cand_locs.fillna("").apply(lambda x: x.str.lower())
        intern_locs = intern_locs.fillna("").str.lower().values
        for pref_idx, score in enumerate([1.0, 0.7, 0.5]):
            cand_pref = cand_locs.iloc[:, pref_idx].values[:, None]
            match_mask = (cand_pref == intern_locs)
            location_matrix[match_mask] = score
        final_matrix = np.clip(0.8 * skill_matrix + 0.2 * location_matrix, 0, 1)
        return final_matrix


from guacamol.standard_benchmarks import isomers_c11h24, isomers_c7h8n2o2, isomers_c9h10n2o2pf2cl, similarity
# https://github.com/BenevolentAI/guacamol/blob/60ebe1f6a396f16e08b834dce448e9343d259feb/guacamol/standard_benchmarks.py#L66


def get_isomer_c7h8n2o2_score(smiles: str):
    return isomers_c7h8n2o2("arithmetic").objective.score(smiles)

def get_isomer_c9h10n2o2pf2cl_score(smiles: str):
    return isomers_c9h10n2o2pf2cl("arithmetic").objective.score(smiles)

def get_albuterol_similarity_score(smiles: str):
    return similarity(smiles='CC(C)(C)NCC(O)c1ccc(O)c(CO)c1', name='Albuterol', fp_type='FCFP4', threshold=0.75).objective.score(smiles)

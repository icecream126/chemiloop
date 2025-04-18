from guacamol.standard_benchmarks import isomers_c11h24, isomers_c7h8n2o2, isomers_c9h10n2o2pf2cl, similarity

# Example SMILES
smiles = "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O"
# https://github.com/BenevolentAI/guacamol/blob/60ebe1f6a396f16e08b834dce448e9343d259feb/guacamol/standard_benchmarks.py#L66

score1 = similarity(smiles='CC(C)(C)NCC(O)c1ccc(O)c(CO)c1', name='Albuterol', fp_type='FCFP4', threshold=0.75).objective.score('CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O')
score2 = isomers_c7h8n2o2("arithmetic").objective.score('CCCCCC')
score3 = isomers_c9h10n2o2pf2cl("arithmetic").objective.score('CCCCCC')


print(score1)
print(score2)
print(score3)
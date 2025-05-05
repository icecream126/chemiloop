from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Fragments
from rdkit import Chem
from rdkit.Chem import Fragments
# ==== Fragments ====
def fr_al_coo(mol): return Fragments.fr_Al_COO(mol, countUnique=True)
def fr_al_oh(mol): return Fragments.fr_Al_OH(mol, countUnique=True)
def fr_al_oh_notert(mol): return Fragments.fr_Al_OH_noTert(mol, countUnique=True)
def fr_arn(mol): return Fragments.fr_ArN(mol, countUnique=True)
def fr_ar_coo(mol): return Fragments.fr_Ar_COO(mol, countUnique=True)
def fr_ar_n(mol): return Fragments.fr_Ar_N(mol, countUnique=True)
def fr_ar_nh(mol): return Fragments.fr_Ar_NH(mol, countUnique=True)
def fr_ar_oh(mol): return Fragments.fr_Ar_OH(mol, countUnique=True)
def fr_coo(mol): return Fragments.fr_COO(mol, countUnique=True)
def fr_coo2(mol): return Fragments.fr_COO2(mol, countUnique=True)
def fr_c_o(mol): return Fragments.fr_C_O(mol, countUnique=True)
def fr_c_o_nocoo(mol): return Fragments.fr_C_O_noCOO(mol, countUnique=True)
def fr_c_s(mol): return Fragments.fr_C_S(mol, countUnique=True)
def fr_hoccn(mol): return Fragments.fr_HOCCN(mol, countUnique=True)
def fr_imine(mol): return Fragments.fr_Imine(mol, countUnique=True)
def fr_nh0(mol): return Fragments.fr_NH0(mol, countUnique=True)
def fr_nh1(mol): return Fragments.fr_NH1(mol, countUnique=True)
def fr_nh2(mol): return Fragments.fr_NH2(mol, countUnique=True)
def fr_n_o(mol): return Fragments.fr_N_O(mol, countUnique=True)
def fr_ndealk1(mol): return Fragments.fr_Ndealkylation1(mol, countUnique=True)
def fr_ndealk2(mol): return Fragments.fr_Ndealkylation2(mol, countUnique=True)
def fr_nhpyrrole(mol): return Fragments.fr_Nhpyrrole(mol, countUnique=True)
def fr_sh(mol): return Fragments.fr_SH(mol, countUnique=True)
def fr_aldehyde(mol): return Fragments.fr_aldehyde(mol, countUnique=True)
def fr_alkyl_carbamate(mol): return Fragments.fr_alkyl_carbamate(mol, countUnique=True)
def fr_alkyl_halide(mol): return Fragments.fr_alkyl_halide(mol, countUnique=True)
def fr_allylic_oxid(mol): return Fragments.fr_allylic_oxid(mol, countUnique=True)
def fr_amide(mol): return Fragments.fr_amide(mol, countUnique=True)
def fr_amidine(mol): return Fragments.fr_amidine(mol, countUnique=True)
def fr_aniline(mol): return Fragments.fr_aniline(mol, countUnique=True)
def fr_aryl_methyl(mol): return Fragments.fr_aryl_methyl(mol, countUnique=True)
def fr_azide(mol): return Fragments.fr_azide(mol, countUnique=True)
def fr_azo(mol): return Fragments.fr_azo(mol, countUnique=True)
def fr_barbitur(mol): return Fragments.fr_barbitur(mol, countUnique=True)
def fr_benzene(mol): return Fragments.fr_benzene(mol, countUnique=True)
def fr_benzodiazepine(mol): return Fragments.fr_benzodiazepine(mol, countUnique=True)
def fr_bicyclic(mol): return Fragments.fr_bicyclic(mol, countUnique=True)
def fr_diazo(mol): return Fragments.fr_diazo(mol, countUnique=True)
def fr_dihydropyridine(mol): return Fragments.fr_dihydropyridine(mol, countUnique=True)
def fr_epoxide(mol): return Fragments.fr_epoxide(mol, countUnique=True)
def fr_ester(mol): return Fragments.fr_ester(mol, countUnique=True)
def fr_ether(mol): return Fragments.fr_ether(mol, countUnique=True)
def fr_furan(mol): return Fragments.fr_furan(mol, countUnique=True)
def fr_guanido(mol): return Fragments.fr_guanido(mol, countUnique=True)
def fr_halogen(mol): return Fragments.fr_halogen(mol, countUnique=True)
def fr_hdrzine(mol): return Fragments.fr_hdrzine(mol, countUnique=True)
def fr_hdrzone(mol): return Fragments.fr_hdrzone(mol, countUnique=True)
def fr_imidazole(mol): return Fragments.fr_imidazole(mol, countUnique=True)
def fr_imide(mol): return Fragments.fr_imide(mol, countUnique=True)
def fr_isocyan(mol): return Fragments.fr_isocyan(mol, countUnique=True)
def fr_isothiocyan(mol): return Fragments.fr_isothiocyan(mol, countUnique=True)
def fr_ketone(mol): return Fragments.fr_ketone(mol, countUnique=True)
def fr_ketone_topliss(mol): return Fragments.fr_ketone_Topliss(mol, countUnique=True)
def fr_lactam(mol): return Fragments.fr_lactam(mol, countUnique=True)
def fr_lactone(mol): return Fragments.fr_lactone(mol, countUnique=True)
def fr_methoxy(mol): return Fragments.fr_methoxy(mol, countUnique=True)
def fr_morpholine(mol): return Fragments.fr_morpholine(mol, countUnique=True)
def fr_nitrile(mol): return Fragments.fr_nitrile(mol, countUnique=True)
def fr_nitro(mol): return Fragments.fr_nitro(mol, countUnique=True)
def fr_nitro_arom(mol): return Fragments.fr_nitro_arom(mol, countUnique=True)
def fr_nitro_arom_nonortho(mol): return Fragments.fr_nitro_arom_nonortho(mol, countUnique=True)
def fr_nitroso(mol): return Fragments.fr_nitroso(mol, countUnique=True)
def fr_oxazole(mol): return Fragments.fr_oxazole(mol, countUnique=True)
def fr_oxime(mol): return Fragments.fr_oxime(mol, countUnique=True)
def fr_para_hydroxylation(mol): return Fragments.fr_para_hydroxylation(mol, countUnique=True)
def fr_phenol(mol): return Fragments.fr_phenol(mol, countUnique=True)
def fr_phenol_noorthohbond(mol): return Fragments.fr_phenol_noOrthoHbond(mol, countUnique=True)
def fr_phos_acid(mol): return Fragments.fr_phos_acid(mol, countUnique=True)
def fr_phos_ester(mol): return Fragments.fr_phos_ester(mol, countUnique=True)
def fr_piperdine(mol): return Fragments.fr_piperdine(mol, countUnique=True)
def fr_piperzine(mol): return Fragments.fr_piperzine(mol, countUnique=True)
def fr_priamide(mol): return Fragments.fr_priamide(mol, countUnique=True)
def fr_prisulfonamd(mol): return Fragments.fr_prisulfonamd(mol, countUnique=True)
def fr_pyridine(mol): return Fragments.fr_pyridine(mol, countUnique=True)
def fr_quatn(mol): return Fragments.fr_quatN(mol, countUnique=True)
def fr_sulfide(mol): return Fragments.fr_sulfide(mol, countUnique=True)
def fr_sulfonamd(mol): return Fragments.fr_sulfonamd(mol, countUnique=True)
def fr_sulfone(mol): return Fragments.fr_sulfone(mol, countUnique=True)
def fr_term_acetylene(mol): return Fragments.fr_term_acetylene(mol, countUnique=True)
def fr_tetrazole(mol): return Fragments.fr_tetrazole(mol, countUnique=True)
def fr_thiazole(mol): return Fragments.fr_thiazole(mol, countUnique=True)
def fr_thiocyan(mol): return Fragments.fr_thiocyan(mol, countUnique=True)
def fr_thiophene(mol): return Fragments.fr_thiophene(mol, countUnique=True)
def fr_unbrch_alkane(mol): return Fragments.fr_unbrch_alkane(mol, countUnique=True)
def fr_urea(mol): return Fragments.fr_urea(mol, countUnique=True)


# ==== rdMolDescriptors ====
def bcut2d(mol):
    return rdMolDescriptors.BCUT2D(mol)

def calcautocorr2d(mol, CustomAtomProperty="GasteigerCharges"):
    return rdMolDescriptors.CalcAUTOCORR2D(mol, CustomAtomProperty)

def calcchi0n(mol, force=None):
    return rdMolDescriptors.CalcChi0n(mol, force)

def calcchi0v(mol, force=None):
    return rdMolDescriptors.CalcChi0v(mol, force)

def calcchi1n(mol, force=None):
    return rdMolDescriptors.CalcChi1n(mol, force)

def calcchi1v(mol, force=None):
    return rdMolDescriptors.CalcChi1v(mol, force)

def calcchi2n(mol, force=None):
    return rdMolDescriptors.CalcChi2n(mol, force)

def calcchi2v(mol, force=None):
    return rdMolDescriptors.CalcChi2v(mol, force)

def calcchi3n(mol, force=None):
    return rdMolDescriptors.CalcChi3n(mol, force)

def calcchi3v(mol, force=None):
    return rdMolDescriptors.CalcChi3v(mol, force)

def calcchi4n(mol, force=None):
    return rdMolDescriptors.CalcChi4n(mol, force)

def calcchi4v(mol, force=None):
    return rdMolDescriptors.CalcChi4v(mol, force)

def calccrippendescriptors(mol, includeHs=None, force=None):
    return rdMolDescriptors.CalcCrippenDescriptors(mol, includeHs, force)

def calcexactmolwt(mol, onlyHeavy=None):
    return rdMolDescriptors.CalcExactMolWt(mol, onlyHeavy)

def calcfractioncsp3(mol):
    return rdMolDescriptors.CalcFractionCSP3(mol)

def calckappa1(mol):
    return rdMolDescriptors.CalcKappa1(mol)

def calckappa2(mol):
    return rdMolDescriptors.CalcKappa2(mol)

def calckappa3(mol):
    return rdMolDescriptors.CalcKappa3(mol)

def calclabuteasa(mol, includeHs=None, force=None):
    return rdMolDescriptors.CalcLabuteASA(mol, includeHs, force)

def calcmolformula(mol, separateIsotopes=None, abbreviateHIsotopes=None):
    return rdMolDescriptors.CalcMolFormula(mol, separateIsotopes, abbreviateHIsotopes)

def calcnumaliphaticcarbocycles(mol):
    return rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)

def calcnumaliphaticheterocycles(mol):
    return rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)

def calcnumaliphaticrings(mol):
    return rdMolDescriptors.CalcNumAliphaticRings(mol)

def calcnumamidebonds(mol):
    return rdMolDescriptors.CalcNumAmideBonds(mol)

def calcnumaromaticcarbocycles(mol):
    return rdMolDescriptors.CalcNumAromaticCarbocycles(mol)

def calcnumaromaticheterocycles(mol):
    return rdMolDescriptors.CalcNumAromaticHeterocycles(mol)

def calcnumaromaticrings(mol):
    return rdMolDescriptors.CalcNumAromaticRings(mol)

def calcnumatomstereocenters(mol):
    return rdMolDescriptors.CalcNumAtomStereoCenters(mol)

def calcnumatoms(mol):
    return rdMolDescriptors.CalcNumAtoms(mol)

def calcnumhba(mol):
    return rdMolDescriptors.CalcNumHBA(mol)

def calcnumhbd(mol):
    return rdMolDescriptors.CalcNumHBD(mol)

def calcnumheavyatoms(mol):
    return rdMolDescriptors.CalcNumHeavyAtoms(mol)

def calcnumheteroatoms(mol):
    return rdMolDescriptors.CalcNumHeteroatoms(mol)

def calcnumheterocycles(mol):
    return rdMolDescriptors.CalcNumHeterocycles(mol)

def calcnumlipinskihba(mol):
    return rdMolDescriptors.CalcNumLipinskiHBA(mol)

def calcnumlipinskihbd(mol):
    return rdMolDescriptors.CalcNumLipinskiHBD(mol)

def calcnumrings(mol):
    return rdMolDescriptors.CalcNumRings(mol)

def calcnumrotatablebonds(mol, strict=None):
    return rdMolDescriptors.CalcNumRotatableBonds(mol, strict)

def calcnumsaturatedcarbocycles(mol):
    return rdMolDescriptors.CalcNumSaturatedCarbocycles(mol)

def calcnumsaturatedheterocycles(mol):
    return rdMolDescriptors.CalcNumSaturatedHeterocycles(mol)

def calcnumsaturatedrings(mol):
    return rdMolDescriptors.CalcNumSaturatedRings(mol)

def calcnumunspecifiedatomstereocenters(mol):
    return rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)

def calcoxidationnumbers(mol):
    return rdMolDescriptors.CalcOxidationNumbers(mol)

def calcpbf(mol, confId=None):
    return rdMolDescriptors.CalcPBF(mol, confId)

def calcphi(mol):
    return rdMolDescriptors.CalcPhi(mol)

def getconnectivityinvariants(mol, includeRingMembership=None):
    return rdMolDescriptors.GetConnectivityInvariants(mol, includeRingMembership)

def getfeatureinvariants(mol):
    return rdMolDescriptors.GetFeatureInvariants(mol)

def gethashedatompairfingerprint(
    mol,
    nBits=1024,
    minLength=1,
    maxLength=30,
    fromAtoms=None,
    ignoreAtoms=None,
    atomInvariants=None,
    includeChirality=False,
    use2D=True,
    confId=-1
):
    return list(rdMolDescriptors.GetHashedAtomPairFingerprint(
        mol, nBits, minLength, maxLength,
        fromAtoms, ignoreAtoms, atomInvariants,
        includeChirality, use2D, confId
    ))

def gethashedatompairfingerprintasbitvect(
    mol,
    nBits=1024,
    minLength=1,
    maxLength=30,
    fromAtoms=None,
    ignoreAtoms=None,
    atomInvariants=None,
    nBitsPerEntry=4,
    includeChirality=False,
    use2D=True,
    confId=-1
):
    return list(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
        mol, nBits, minLength, maxLength,
        fromAtoms, ignoreAtoms, atomInvariants,
        nBitsPerEntry, includeChirality, use2D, confId
    ))

def gethashedmorganfingerprint(
    mol,
    radius=2,
    nBits=1024,
    invariants=None,
    fromAtoms=None,
    useChirality=False,
    useBondTypes=True,
    useFeatures=False,
    bitInfo=None,
    includeRedundantEnvironments=False
):
    return list(rdMolDescriptors.GetHashedMorganFingerprint(
        mol, radius, nBits, invariants, fromAtoms,
        useChirality, useBondTypes, useFeatures,
        bitInfo, includeRedundantEnvironments
    ))

def gethashedtopologicaltorsionfingerprint(
    mol,
    nBits=1024,
    targetSize=4,
    fromAtoms=None,
    ignoreAtoms=None,
    atomInvariants=None,
    includeChirality=False
):
    return list(rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(
        mol, nBits, targetSize, fromAtoms,
        ignoreAtoms, atomInvariants, includeChirality
    ))

def gethashedtopologicaltorsionfingerprintasbitvect(
    mol,
    nBits=1024,
    targetSize=4,
    fromAtoms=None,
    ignoreAtoms=None,
    atomInvariants=None,
    nBitsPerEntry=4,
    includeChirality=False
):
    return list(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
        mol, nBits, targetSize, fromAtoms,
        ignoreAtoms, atomInvariants,
        nBitsPerEntry, includeChirality
    ))

def getmaccskeysfingerprint(mol):
    fp=rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    return list(fp.GetOnBits())

def getmorganfingerprint(mol, radius=2):
    fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=2)
    return fp.GetNonzeroElements()

def getmorganfingerprintasbitvect(
    mol,
    radius=2,
    nBits=1024,
    invariants=None,
    fromAtoms=None,
    useChirality=False,
    useBondTypes=True,
    useFeatures=False,
    bitInfo=None,
    includeRedundantEnvironments=False
):
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return list(fp.GetOnBits())# , nBits, invariants, fromAtoms, useChirality, useBondTypes, useFeatures, bitInfo, includeRedundantEnvironments))

def gettopologicaltorsionfingerprint(
    mol,
    targetSize=4,
    fromAtoms=None,
    ignoreAtoms=None,
    atomInvariants=None,
    includeChirality=False
):
    fp = rdMolDescriptors.GetTopologicalTorsionFingerprint(mol, targetSize=4)
    return fp.GetNonzeroElements()

def mqns_(mol, force=None):
    return rdMolDescriptors.MQNs_(mol, force)

def peoe_vsa_(mol, bins=None, force=None):
    return rdMolDescriptors.PEOE_VSA_(mol, bins, force)

def smr_vsa_(mol, bins=None, force=None):
    return rdMolDescriptors.SMR_VSA_(mol, bins, force)

def slogp_vsa_(mol, bins=None, force=None):
    return rdMolDescriptors.SlogP_VSA_(mol, bins, force)

# if __name__=="__main__":

#     import inspect
#     import sys

#     # Create test mol
#     mol = Chem.MolFromSmiles("CCCCCC")

#     import pdb; pdb.set_trace()
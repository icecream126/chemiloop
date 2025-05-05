from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Fragments

# ==== rdMolDescriptors ====
def bcut2d(mol):
    return list(rdMolDescriptors.BCUT2D(mol))

def calcautocorr2d(mol, CustomAtomProperty="GasteigerCharges"):
    return list(rdMolDescriptors.CalcAUTOCORR2D(mol, CustomAtomProperty))

def calcchi0n(mol, force=None):
    return list(rdMolDescriptors.CalcChi0n(mol, force))

def calcchi0v(mol, force=None):
    return list(rdMolDescriptors.CalcChi0v(mol, force))

def calcchi1n(mol, force=None):
    return list(rdMolDescriptors.CalcChi1n(mol, force))

def calcchi1v(mol, force=None):
    return list(rdMolDescriptors.CalcChi1v(mol, force))

def calcchi2n(mol, force=None):
    return list(rdMolDescriptors.CalcChi2n(mol, force))

def calcchi2v(mol, force=None):
    return list(rdMolDescriptors.CalcChi2v(mol, force))

def calcchi3n(mol, force=None):
    return list(rdMolDescriptors.CalcChi3n(mol, force))

def calcchi3v(mol, force=None):
    return list(rdMolDescriptors.CalcChi3v(mol, force))

def calcchi4n(mol, force=None):
    return list(rdMolDescriptors.CalcChi4n(mol, force))

def calcchi4v(mol, force=None):
    return list(rdMolDescriptors.CalcChi4v(mol, force))

def calccrippendescriptors(mol, includeHs=None, force=None):
    return list(rdMolDescriptors.CalcCrippenDescriptors(mol, includeHs, force))

def calcexactmolwt(mol, onlyHeavy=None):
    return list(rdMolDescriptors.CalcExactMolWt(mol, onlyHeavy))

def calcfractioncsp3(mol):
    return list(rdMolDescriptors.CalcFractionCSP3(mol))

def calckappa1(mol):
    return list(rdMolDescriptors.CalcKappa1(mol))

def calckappa2(mol):
    return list(rdMolDescriptors.CalcKappa2(mol))

def calckappa3(mol):
    return list(rdMolDescriptors.CalcKappa3(mol))

def calclabuteasa(mol, includeHs=None, force=None):
    return list(rdMolDescriptors.CalcLabuteASA(mol, includeHs, force))

def calcmolformula(mol, separateIsotopes=None, abbreviateHIsotopes=None):
    return list(rdMolDescriptors.CalcMolFormula(mol, separateIsotopes, abbreviateHIsotopes))

def calcnumaliphaticcarbocycles(mol):
    return list(rdMolDescriptors.CalcNumAliphaticCarbocycles(mol))

def calcnumaliphaticheterocycles(mol):
    return list(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))

def calcnumaliphaticrings(mol):
    return list(rdMolDescriptors.CalcNumAliphaticRings(mol))

def calcnumamidebonds(mol):
    return list(rdMolDescriptors.CalcNumAmideBonds(mol))

def calcnumaromaticcarbocycles(mol):
    return list(rdMolDescriptors.CalcNumAromaticCarbocycles(mol))

def calcnumaromaticheterocycles(mol):
    return list(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))

def calcnumaromaticrings(mol):
    return list(rdMolDescriptors.CalcNumAromaticRings(mol))

def calcnumatomstereocenters(mol):
    return list(rdMolDescriptors.CalcNumAtomStereoCenters(mol))

def calcnumatoms(mol):
    return list(rdMolDescriptors.CalcNumAtoms(mol))

def calcnumhba(mol):
    return list(rdMolDescriptors.CalcNumHBA(mol))

def calcnumhbd(mol):
    return list(rdMolDescriptors.CalcNumHBD(mol))

def calcnumheavyatoms(mol):
    return list(rdMolDescriptors.CalcNumHeavyAtoms(mol))

def calcnumheteroatoms(mol):
    return list(rdMolDescriptors.CalcNumHeteroatoms(mol))

def calcnumheterocycles(mol):
    return list(rdMolDescriptors.CalcNumHeterocycles(mol))

def calcnumlipinskihba(mol):
    return list(rdMolDescriptors.CalcNumLipinskiHBA(mol))

def calcnumlipinskihbd(mol):
    return list(rdMolDescriptors.CalcNumLipinskiHBD(mol))

def calcnumrings(mol):
    return list(rdMolDescriptors.CalcNumRings(mol))

def calcnumrotatablebonds(mol, strict=None):
    return list(rdMolDescriptors.CalcNumRotatableBonds(mol, strict))

def calcnumsaturatedcarbocycles(mol):
    return list(rdMolDescriptors.CalcNumSaturatedCarbocycles(mol))

def calcnumsaturatedheterocycles(mol):
    return list(rdMolDescriptors.CalcNumSaturatedHeterocycles(mol))

def calcnumsaturatedrings(mol):
    return list(rdMolDescriptors.CalcNumSaturatedRings(mol))

def calcnumunspecifiedatomstereocenters(mol):
    return list(rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol))

def calcoxidationnumbers(mol):
    return list(rdMolDescriptors.CalcOxidationNumbers(mol))

def calcpbf(mol, confId=None):
    return list(rdMolDescriptors.CalcPBF(mol, confId))

def calcphi(mol):
    return list(rdMolDescriptors.CalcPhi(mol))

def getconnectivityinvariants(mol, includeRingMembership=None):
    return list(rdMolDescriptors.GetConnectivityInvariants(mol, includeRingMembership))

def getfeatureinvariants(mol):
    return list(rdMolDescriptors.GetFeatureInvariants(mol))

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
    return list(rdMolDescriptors.GetMACCSKeysFingerprint(mol))

def getmorganfingerprint(mol, radius=2):
    return list(rdMolDescriptors.GetMorganFingerprint(mol, radius))

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
    return list(rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius, nBits, invariants, fromAtoms,
        useChirality, useBondTypes, useFeatures,
        bitInfo, includeRedundantEnvironments
    ))

def gettopologicaltorsionfingerprint(
    mol,
    targetSize=4,
    fromAtoms=None,
    ignoreAtoms=None,
    atomInvariants=None,
    includeChirality=False
):
    return list(rdMolDescriptors.GetTopologicalTorsionFingerprint(
        mol, targetSize, fromAtoms,
        ignoreAtoms, atomInvariants, includeChirality
    ))

def mqns_(mol, force=None):
    return list(rdMolDescriptors.MQNs_(mol, force))

def peoe_vsa_(mol, bins=None, force=None):
    return list(rdMolDescriptors.PEOE_VSA_(mol, bins, force))

def smr_vsa_(mol, bins=None, force=None):
    return list(rdMolDescriptors.SMR_VSA_(mol, bins, force))

def slogp_vsa_(mol, bins=None, force=None):
    return list(rdMolDescriptors.SlogP_VSA_(mol, bins, force))



import inspect
import sys

# Create test mol
mol = Chem.MolFromSmiles("CCCCCC")

import pdb; pdb.set_trace()
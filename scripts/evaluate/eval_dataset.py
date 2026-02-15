"""
Evaluation Dataset for GraphRAG

Contains 100+ questions across 9 Singapore government policy categories.
Questions are categorized by:
- Category: CPF, HDB, IRAS, MOH, MOM, MSF, Grants, SkillsFuture, Education
- Type: specific (factual), multi-hop (reasoning), overview (broad)
- Difficulty: easy, medium, hard

Each question includes:
- Expected entities that should be retrieved
- Keywords that should appear in a correct answer
- Reference source (for verification)
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import json


class Category(str, Enum):
    CPF = "cpf"
    HDB = "hdb"
    IRAS = "iras"
    MOH = "moh"
    MOM = "mom"
    MSF = "msf"
    GRANTS = "grants"
    SKILLSFUTURE = "skillsfuture"
    EDUCATION = "education"


class QuestionType(str, Enum):
    SPECIFIC = "specific"      # Single fact lookup
    MULTI_HOP = "multi_hop"    # Requires connecting multiple entities
    OVERVIEW = "overview"      # Broad summary/comparison
    ELIGIBILITY = "eligibility"  # "Am I eligible for X?" type
    PROCESS = "process"        # "How do I apply for X?" type
    COMPARISON = "comparison"  # Compare two schemes/options


class Difficulty(str, Enum):
    EASY = "easy"      # Direct lookup, single entity
    MEDIUM = "medium"  # Some reasoning, 2-3 entities
    HARD = "hard"      # Complex reasoning, multiple hops


@dataclass
class EvalQuestion:
    """A single evaluation question with ground truth."""
    id: str
    question: str
    category: Category
    question_type: QuestionType
    difficulty: Difficulty
    expected_entities: list[str]  # Entity names that should be retrieved
    answer_keywords: list[str]    # Keywords that should appear in answer
    reference_note: str = ""      # Optional note about source/verification

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "category": self.category.value,
            "question_type": self.question_type.value,
            "difficulty": self.difficulty.value,
            "expected_entities": self.expected_entities,
            "answer_keywords": self.answer_keywords,
            "reference_note": self.reference_note,
        }


# =============================================================================
# CPF Questions (Central Provident Fund)
# =============================================================================

CPF_QUESTIONS = [
    EvalQuestion(
        id="cpf_001",
        question="What are the four CPF accounts and their purposes?",
        category=Category.CPF,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.EASY,
        expected_entities=["cpf", "ordinary_account", "special_account", "medisave_account", "retirement_account"],
        answer_keywords=["Ordinary Account", "Special Account", "MediSave", "Retirement Account", "housing", "retirement", "healthcare"],
    ),
    EvalQuestion(
        id="cpf_002",
        question="What is the current CPF contribution rate for employees below 55 years old?",
        category=Category.CPF,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.EASY,
        expected_entities=["cpf_contribution", "contribution_rate", "employee"],
        answer_keywords=["20%", "17%", "37%", "employer", "employee"],
    ),
    EvalQuestion(
        id="cpf_003",
        question="How can I use my CPF Ordinary Account for housing?",
        category=Category.CPF,
        question_type=QuestionType.PROCESS,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["ordinary_account", "housing", "hdb", "downpayment", "mortgage"],
        answer_keywords=["downpayment", "monthly instalments", "HDB", "private property", "accrued interest"],
    ),
    EvalQuestion(
        id="cpf_004",
        question="What is the Basic Retirement Sum (BRS) and how is it calculated?",
        category=Category.CPF,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["basic_retirement_sum", "full_retirement_sum", "enhanced_retirement_sum", "retirement_account"],
        answer_keywords=["BRS", "FRS", "ERS", "CPF LIFE", "property pledge"],
    ),
    EvalQuestion(
        id="cpf_005",
        question="What is CPF LIFE and when does it start paying out?",
        category=Category.CPF,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["cpf_life", "retirement_account", "payout", "annuity"],
        answer_keywords=["lifelong", "monthly", "65", "payout age", "annuity"],
    ),
    EvalQuestion(
        id="cpf_006",
        question="Can I withdraw my CPF savings if I am leaving Singapore permanently?",
        category=Category.CPF,
        question_type=QuestionType.ELIGIBILITY,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["cpf_withdrawal", "emigration", "permanent_departure"],
        answer_keywords=["withdraw", "full", "renounce", "citizenship", "PR"],
    ),
    EvalQuestion(
        id="cpf_007",
        question="How does the CPF Housing Grant work with my CPF OA for buying a BTO flat?",
        category=Category.CPF,
        question_type=QuestionType.MULTI_HOP,
        difficulty=Difficulty.HARD,
        expected_entities=["cpf_housing_grant", "ordinary_account", "bto", "enhanced_housing_grant", "first_timer"],
        answer_keywords=["EHG", "income ceiling", "first-timer", "downpayment", "OA"],
    ),
    EvalQuestion(
        id="cpf_008",
        question="What is the CPF Annual Limit and how does it affect my contributions?",
        category=Category.CPF,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["cpf_annual_limit", "contribution", "ordinary_wage_ceiling"],
        answer_keywords=["$37,740", "annual", "ceiling", "ordinary wages"],
    ),
    EvalQuestion(
        id="cpf_009",
        question="How can I top up my CPF for tax relief?",
        category=Category.CPF,
        question_type=QuestionType.PROCESS,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["cpf_top_up", "tax_relief", "retirement_sum_topping_up", "special_account"],
        answer_keywords=["RSTU", "tax relief", "$8,000", "$16,000", "loved ones"],
    ),
    EvalQuestion(
        id="cpf_010",
        question="What happens to my CPF if I pass away?",
        category=Category.CPF,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.EASY,
        expected_entities=["cpf_nomination", "beneficiary", "cpf_board"],
        answer_keywords=["nomination", "beneficiaries", "Public Trustee", "distribute"],
    ),
]

# =============================================================================
# HDB Questions (Housing Development Board)
# =============================================================================

HDB_QUESTIONS = [
    EvalQuestion(
        id="hdb_001",
        question="What is the income ceiling for buying a BTO flat?",
        category=Category.HDB,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.EASY,
        expected_entities=["bto", "income_ceiling", "eligibility"],
        answer_keywords=["$14,000", "household income", "gross monthly"],
    ),
    EvalQuestion(
        id="hdb_002",
        question="What is the Minimum Occupation Period (MOP) for HDB flats?",
        category=Category.HDB,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.EASY,
        expected_entities=["minimum_occupation_period", "mop", "hdb"],
        answer_keywords=["5 years", "MOP", "sell", "rent out"],
    ),
    EvalQuestion(
        id="hdb_003",
        question="What grants are available for first-time HDB buyers?",
        category=Category.HDB,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["enhanced_cpf_housing_grant", "family_grant", "proximity_housing_grant", "first_timer"],
        answer_keywords=["EHG", "Family Grant", "Proximity Grant", "first-timer", "income"],
    ),
    EvalQuestion(
        id="hdb_004",
        question="Can singles buy an HDB flat? What are the requirements?",
        category=Category.HDB,
        question_type=QuestionType.ELIGIBILITY,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["single_singapore_citizen", "hdb_eligibility", "2_room_flexi"],
        answer_keywords=["35 years old", "Single Singapore Citizen", "2-room Flexi", "resale"],
    ),
    EvalQuestion(
        id="hdb_005",
        question="What is the Enhanced CPF Housing Grant (EHG) and who is eligible?",
        category=Category.HDB,
        question_type=QuestionType.ELIGIBILITY,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["enhanced_cpf_housing_grant", "income_ceiling", "first_timer"],
        answer_keywords=["$80,000", "first-timer", "income ceiling", "$9,000"],
    ),
    EvalQuestion(
        id="hdb_006",
        question="How does the resale levy work when buying a second subsidised flat?",
        category=Category.HDB,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.HARD,
        expected_entities=["resale_levy", "second_timer", "subsidised_flat", "bto"],
        answer_keywords=["resale levy", "second subsidised", "flat type", "selling price"],
    ),
    EvalQuestion(
        id="hdb_007",
        question="What is the process for applying for a BTO flat?",
        category=Category.HDB,
        question_type=QuestionType.PROCESS,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["bto", "application", "ballot", "queue_number"],
        answer_keywords=["HDB portal", "ballot", "queue number", "book flat", "sign agreement"],
    ),
    EvalQuestion(
        id="hdb_008",
        question="Can I rent out my entire HDB flat?",
        category=Category.HDB,
        question_type=QuestionType.ELIGIBILITY,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["subletting", "mop", "hdb_rental"],
        answer_keywords=["MOP", "approval", "minimum 6 months", "non-citizen quota"],
    ),
    EvalQuestion(
        id="hdb_009",
        question="What is the Proximity Housing Grant?",
        category=Category.HDB,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.EASY,
        expected_entities=["proximity_housing_grant", "parents", "married_child"],
        answer_keywords=["$30,000", "$20,000", "parents", "4km", "live together"],
    ),
    EvalQuestion(
        id="hdb_010",
        question="If I earn $10,000/month and want to buy a BTO, what grants am I eligible for?",
        category=Category.HDB,
        question_type=QuestionType.MULTI_HOP,
        difficulty=Difficulty.HARD,
        expected_entities=["bto", "income_ceiling", "enhanced_cpf_housing_grant", "first_timer"],
        answer_keywords=["eligible", "EHG", "income ceiling", "$14,000"],
    ),
]

# =============================================================================
# IRAS Questions (Tax)
# =============================================================================

IRAS_QUESTIONS = [
    EvalQuestion(
        id="iras_001",
        question="What is the personal income tax rate in Singapore?",
        category=Category.IRAS,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.EASY,
        expected_entities=["personal_income_tax", "tax_rate", "progressive_tax"],
        answer_keywords=["progressive", "0%", "22%", "24%", "chargeable income"],
    ),
    EvalQuestion(
        id="iras_002",
        question="What tax reliefs are available for CPF contributions?",
        category=Category.IRAS,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["cpf_relief", "tax_relief", "cpf_top_up"],
        answer_keywords=["CPF Relief", "RSTU", "$8,000", "compulsory contributions"],
    ),
    EvalQuestion(
        id="iras_003",
        question="How do I claim tax relief for course fees?",
        category=Category.IRAS,
        question_type=QuestionType.PROCESS,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["course_fee_relief", "tax_relief", "self_improvement"],
        answer_keywords=["$5,500", "approved course", "degree", "professional qualification"],
    ),
    EvalQuestion(
        id="iras_004",
        question="What is the GST rate in Singapore and when does it apply?",
        category=Category.IRAS,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.EASY,
        expected_entities=["gst", "goods_and_services_tax", "gst_rate"],
        answer_keywords=["9%", "goods", "services", "exempt", "zero-rated"],
    ),
    EvalQuestion(
        id="iras_005",
        question="What are the property tax rates for owner-occupied homes?",
        category=Category.IRAS,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["property_tax", "owner_occupied", "annual_value"],
        answer_keywords=["owner-occupied", "progressive", "Annual Value", "0%", "first $8,000"],
    ),
    EvalQuestion(
        id="iras_006",
        question="When is the tax filing deadline in Singapore?",
        category=Category.IRAS,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.EASY,
        expected_entities=["tax_filing", "deadline", "form_b"],
        answer_keywords=["18 April", "15 April", "e-filing", "paper filing"],
    ),
    EvalQuestion(
        id="iras_007",
        question="How can I reduce my taxable income legally?",
        category=Category.IRAS,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["tax_relief", "deduction", "cpf_top_up", "srs"],
        answer_keywords=["CPF top-up", "SRS", "donations", "course fees", "NSman relief"],
    ),
    EvalQuestion(
        id="iras_008",
        question="What is the Supplementary Retirement Scheme (SRS) and how does it help with taxes?",
        category=Category.IRAS,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["supplementary_retirement_scheme", "srs", "tax_relief"],
        answer_keywords=["SRS", "$15,300", "tax deduction", "retirement", "withdraw"],
    ),
    EvalQuestion(
        id="iras_009",
        question="Are there tax reliefs for parents supporting their children's education?",
        category=Category.IRAS,
        question_type=QuestionType.ELIGIBILITY,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["parent_relief", "qualifying_child_relief", "tax_relief"],
        answer_keywords=["Qualifying Child Relief", "$4,000", "unmarried child", "full-time education"],
    ),
    EvalQuestion(
        id="iras_010",
        question="How is rental income taxed in Singapore?",
        category=Category.IRAS,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["rental_income", "property_tax", "income_tax"],
        answer_keywords=["taxable income", "deductions", "property tax", "maintenance", "interest"],
    ),
]

# =============================================================================
# MOH Questions (Healthcare)
# =============================================================================

MOH_QUESTIONS = [
    EvalQuestion(
        id="moh_001",
        question="What is MediShield Life and what does it cover?",
        category=Category.MOH,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.EASY,
        expected_entities=["medishield_life", "basic_health_insurance", "hospitalisation"],
        answer_keywords=["basic health insurance", "hospitalisation", "B2/C ward", "lifetime", "pre-existing"],
    ),
    EvalQuestion(
        id="moh_002",
        question="How do CHAS subsidies work and who is eligible?",
        category=Category.MOH,
        question_type=QuestionType.ELIGIBILITY,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["chas", "community_health_assist_scheme", "subsidy", "blue_card", "orange_card"],
        answer_keywords=["Blue", "Orange", "Green", "household income", "clinic", "dental"],
    ),
    EvalQuestion(
        id="moh_003",
        question="What is MediSave and what can it be used for?",
        category=Category.MOH,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.EASY,
        expected_entities=["medisave", "cpf", "healthcare", "hospitalisation"],
        answer_keywords=["CPF", "hospitalisation", "outpatient", "MediShield Life", "Integrated Shield Plan"],
    ),
    EvalQuestion(
        id="moh_004",
        question="What is CareShield Life and when does it start?",
        category=Category.MOH,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["careshield_life", "eldershield", "severe_disability", "long_term_care"],
        answer_keywords=["severe disability", "long-term care", "2020", "lifetime payouts", "activities of daily living"],
    ),
    EvalQuestion(
        id="moh_005",
        question="What benefits do Pioneer Generation seniors receive?",
        category=Category.MOH,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["pioneer_generation", "pg_benefits", "medisave_top_up", "outpatient_subsidy"],
        answer_keywords=["Pioneer Generation", "MediSave top-up", "outpatient", "MediShield Life", "subsidies"],
    ),
    EvalQuestion(
        id="moh_006",
        question="How do I apply for Medifund assistance?",
        category=Category.MOH,
        question_type=QuestionType.PROCESS,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["medifund", "financial_assistance", "means_test"],
        answer_keywords=["Medifund", "medical social worker", "subsidised bills", "unable to pay"],
    ),
    EvalQuestion(
        id="moh_007",
        question="What is the difference between MediShield Life and Integrated Shield Plans?",
        category=Category.MOH,
        question_type=QuestionType.COMPARISON,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["medishield_life", "integrated_shield_plan", "private_insurer"],
        answer_keywords=["basic", "additional coverage", "private hospital", "A ward", "riders"],
    ),
    EvalQuestion(
        id="moh_008",
        question="How does the Healthier SG programme work?",
        category=Category.MOH,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["healthier_sg", "family_doctor", "preventive_care"],
        answer_keywords=["enrol", "family doctor", "preventive care", "chronic conditions", "health plan"],
    ),
    EvalQuestion(
        id="moh_009",
        question="What Merdeka Generation benefits are available?",
        category=Category.MOH,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["merdeka_generation", "medisave_top_up", "outpatient_subsidy"],
        answer_keywords=["Merdeka Generation", "MediSave", "outpatient", "CHAS", "subsidies"],
    ),
    EvalQuestion(
        id="moh_010",
        question="Can I use MediSave to pay for my family's medical expenses?",
        category=Category.MOH,
        question_type=QuestionType.ELIGIBILITY,
        difficulty=Difficulty.EASY,
        expected_entities=["medisave", "family", "dependants"],
        answer_keywords=["spouse", "children", "parents", "grandparents", "siblings"],
    ),
]

# =============================================================================
# MOM Questions (Employment/Manpower)
# =============================================================================

MOM_QUESTIONS = [
    EvalQuestion(
        id="mom_001",
        question="What are the different types of work passes in Singapore?",
        category=Category.MOM,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.EASY,
        expected_entities=["employment_pass", "s_pass", "work_permit", "one_pass"],
        answer_keywords=["Employment Pass", "S Pass", "Work Permit", "ONE Pass", "salary"],
    ),
    EvalQuestion(
        id="mom_002",
        question="What is the minimum salary for an Employment Pass?",
        category=Category.MOM,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.EASY,
        expected_entities=["employment_pass", "minimum_salary", "qualifying_salary"],
        answer_keywords=["$5,000", "financial services", "experience", "COMPASS"],
    ),
    EvalQuestion(
        id="mom_003",
        question="What is the retirement age in Singapore?",
        category=Category.MOM,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.EASY,
        expected_entities=["retirement_age", "re_employment", "senior_worker"],
        answer_keywords=["63", "re-employment", "68", "Retirement Age"],
    ),
    EvalQuestion(
        id="mom_004",
        question="How many days of annual leave are employees entitled to?",
        category=Category.MOM,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.EASY,
        expected_entities=["annual_leave", "employment_act", "leave_entitlement"],
        answer_keywords=["7 days", "14 days", "years of service", "Employment Act"],
    ),
    EvalQuestion(
        id="mom_005",
        question="What is the COMPASS framework for Employment Pass applications?",
        category=Category.MOM,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["compass", "employment_pass", "points_based"],
        answer_keywords=["COMPASS", "points", "salary", "qualifications", "diversity", "40 points"],
    ),
    EvalQuestion(
        id="mom_006",
        question="How much maternity leave are working mothers entitled to?",
        category=Category.MOM,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.EASY,
        expected_entities=["maternity_leave", "working_mother", "employment_act"],
        answer_keywords=["16 weeks", "12 weeks", "employer-paid", "government-paid"],
    ),
    EvalQuestion(
        id="mom_007",
        question="What is the foreign worker levy and how is it calculated?",
        category=Category.MOM,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["foreign_worker_levy", "work_permit", "dependency_ratio"],
        answer_keywords=["levy", "tier", "sector", "dependency ratio ceiling"],
    ),
    EvalQuestion(
        id="mom_008",
        question="What protections exist for workplace injuries?",
        category=Category.MOM,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["work_injury_compensation", "wica", "workplace_safety"],
        answer_keywords=["WICA", "medical leave", "compensation", "permanent incapacity", "medical expenses"],
    ),
    EvalQuestion(
        id="mom_009",
        question="Can Employment Pass holders bring their family to Singapore?",
        category=Category.MOM,
        question_type=QuestionType.ELIGIBILITY,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["employment_pass", "dependant_pass", "long_term_visit_pass"],
        answer_keywords=["Dependant's Pass", "$6,000", "spouse", "children", "Long Term Visit Pass"],
    ),
    EvalQuestion(
        id="mom_010",
        question="What is the Progressive Wage Model?",
        category=Category.MOM,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["progressive_wage_model", "pwm", "minimum_wage"],
        answer_keywords=["PWM", "cleaning", "security", "landscape", "skills ladder", "wages"],
    ),
]

# =============================================================================
# MSF Questions (Social Services)
# =============================================================================

MSF_QUESTIONS = [
    EvalQuestion(
        id="msf_001",
        question="What is ComCare and who is eligible?",
        category=Category.MSF,
        question_type=QuestionType.ELIGIBILITY,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["comcare", "financial_assistance", "social_service_office"],
        answer_keywords=["ComCare", "low-income", "SSO", "financial assistance", "household income"],
    ),
    EvalQuestion(
        id="msf_002",
        question="What is the Silver Support Scheme?",
        category=Category.MSF,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.EASY,
        expected_entities=["silver_support_scheme", "elderly", "cash_supplement"],
        answer_keywords=["Silver Support", "65", "quarterly", "cash", "seniors", "low income"],
    ),
    EvalQuestion(
        id="msf_003",
        question="How do CDC Vouchers work?",
        category=Category.MSF,
        question_type=QuestionType.PROCESS,
        difficulty=Difficulty.EASY,
        expected_entities=["cdc_vouchers", "community_development_council", "household"],
        answer_keywords=["CDC Vouchers", "Singpass", "heartland", "supermarkets", "hawkers"],
    ),
    EvalQuestion(
        id="msf_004",
        question="What support is available for caregivers of persons with disabilities?",
        category=Category.MSF,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["caregiver_support", "disability", "respite_care"],
        answer_keywords=["caregiver", "respite", "training", "subsidies", "SG Enable"],
    ),
    EvalQuestion(
        id="msf_005",
        question="How can I apply for short-term financial assistance?",
        category=Category.MSF,
        question_type=QuestionType.PROCESS,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["comcare_smta", "short_term_assistance", "social_service_office"],
        answer_keywords=["SMTA", "SSO", "SupportGoWhere", "temporary", "3-6 months"],
    ),
    EvalQuestion(
        id="msf_006",
        question="What services are available for seniors in Singapore?",
        category=Category.MSF,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["eldercare", "senior_services", "aic"],
        answer_keywords=["eldercare", "day care", "home care", "AIC", "nursing home"],
    ),
    EvalQuestion(
        id="msf_007",
        question="How does the adoption process work in Singapore?",
        category=Category.MSF,
        question_type=QuestionType.PROCESS,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["adoption", "msf", "child_protection"],
        answer_keywords=["adoption", "MSF", "home study", "eligibility", "court order"],
    ),
    EvalQuestion(
        id="msf_008",
        question="What is the Assistive Technology Fund?",
        category=Category.MSF,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["assistive_technology_fund", "disability", "subsidy"],
        answer_keywords=["Assistive Technology", "disability", "devices", "subsidy", "90%"],
    ),
    EvalQuestion(
        id="msf_009",
        question="What is ComCare Long-Term Assistance?",
        category=Category.MSF,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["comcare_lta", "long_term_assistance", "permanently_unable"],
        answer_keywords=["LTA", "permanently unable to work", "elderly", "disability", "monthly cash"],
    ),
    EvalQuestion(
        id="msf_010",
        question="What family support services are available for divorce?",
        category=Category.MSF,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["divorce_support", "family_service_centre", "mediation"],
        answer_keywords=["divorce", "counselling", "mediation", "Family Service Centre", "children"],
    ),
]

# =============================================================================
# Grants Questions (Business Grants)
# =============================================================================

GRANTS_QUESTIONS = [
    EvalQuestion(
        id="grants_001",
        question="What is the Productivity Solutions Grant (PSG)?",
        category=Category.GRANTS,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.EASY,
        expected_entities=["productivity_solutions_grant", "psg", "digital_solutions"],
        answer_keywords=["PSG", "50%", "digital solutions", "equipment", "SME"],
    ),
    EvalQuestion(
        id="grants_002",
        question="What grants are available for SMEs going digital?",
        category=Category.GRANTS,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["psg", "sme_go_digital", "edg", "imda"],
        answer_keywords=["PSG", "SME Go Digital", "IMDA", "digital", "solutions"],
    ),
    EvalQuestion(
        id="grants_003",
        question="What is the Enterprise Development Grant (EDG)?",
        category=Category.GRANTS,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["enterprise_development_grant", "edg", "enterprise_singapore"],
        answer_keywords=["EDG", "50%", "70%", "core capabilities", "innovation", "overseas expansion"],
    ),
    EvalQuestion(
        id="grants_004",
        question="How do I apply for business grants through the Business Grants Portal?",
        category=Category.GRANTS,
        question_type=QuestionType.PROCESS,
        difficulty=Difficulty.EASY,
        expected_entities=["business_grants_portal", "gobusiness", "corppass"],
        answer_keywords=["Business Grants Portal", "CorpPass", "GoBusiness", "apply", "eligibility"],
    ),
    EvalQuestion(
        id="grants_005",
        question="What is the Market Readiness Assistance (MRA) Grant?",
        category=Category.GRANTS,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["market_readiness_assistance", "mra", "overseas_expansion"],
        answer_keywords=["MRA", "overseas", "market", "$100,000", "internationalisation"],
    ),
    EvalQuestion(
        id="grants_006",
        question="What support is available for startups in Singapore?",
        category=Category.GRANTS,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["startup_sg", "startup_sg_founder", "startup_sg_tech"],
        answer_keywords=["Startup SG", "Founder", "Tech", "Equity", "mentorship", "funding"],
    ),
    EvalQuestion(
        id="grants_007",
        question="What is the SkillsFuture Enterprise Credit (SFEC)?",
        category=Category.GRANTS,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["skillsfuture_enterprise_credit", "sfec", "transformation"],
        answer_keywords=["SFEC", "$10,000", "transformation", "workforce", "training"],
    ),
    EvalQuestion(
        id="grants_008",
        question="What are the eligibility criteria for PSG?",
        category=Category.GRANTS,
        question_type=QuestionType.ELIGIBILITY,
        difficulty=Difficulty.EASY,
        expected_entities=["psg", "eligibility", "sme"],
        answer_keywords=["registered", "Singapore", "30%", "local shareholding", "group revenue"],
    ),
]

# =============================================================================
# SkillsFuture Questions
# =============================================================================

SKILLSFUTURE_QUESTIONS = [
    EvalQuestion(
        id="sf_001",
        question="What is SkillsFuture Credit and how much do I get?",
        category=Category.SKILLSFUTURE,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.EASY,
        expected_entities=["skillsfuture_credit", "training", "subsidy"],
        answer_keywords=["$500", "opening credit", "top-up", "25 years old", "eligible courses"],
    ),
    EvalQuestion(
        id="sf_002",
        question="What is the SkillsFuture Mid-Career Enhanced Subsidy?",
        category=Category.SKILLSFUTURE,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["mid_career_enhanced_subsidy", "skillsfuture", "40_years_old"],
        answer_keywords=["40 years old", "90%", "course fees", "mid-career"],
    ),
    EvalQuestion(
        id="sf_003",
        question="How do Career Conversion Programmes (CCPs) work?",
        category=Category.SKILLSFUTURE,
        question_type=QuestionType.PROCESS,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["career_conversion_programme", "ccp", "wsg", "place_and_train"],
        answer_keywords=["CCP", "WSG", "salary support", "training", "new industry", "Place-and-Train"],
    ),
    EvalQuestion(
        id="sf_004",
        question="What is the SkillsFuture Level-Up Programme?",
        category=Category.SKILLSFUTURE,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["skillsfuture_level_up", "additional_credit", "40_years_old"],
        answer_keywords=["Level-Up", "$4,000", "40 years old", "diploma", "degree"],
    ),
    EvalQuestion(
        id="sf_005",
        question="What courses can I use SkillsFuture Credit for?",
        category=Category.SKILLSFUTURE,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.EASY,
        expected_entities=["skillsfuture_credit", "eligible_courses", "ssg"],
        answer_keywords=["SSG", "approved", "academic", "professional", "MySkillsFuture"],
    ),
    EvalQuestion(
        id="sf_006",
        question="What is the SkillsFuture Career Transition Programme?",
        category=Category.SKILLSFUTURE,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["skillsfuture_career_transition_programme", "sctp", "mid_career"],
        answer_keywords=["SCTP", "mid-career", "train-and-place", "industry attachment"],
    ),
    EvalQuestion(
        id="sf_007",
        question="How can employers benefit from SkillsFuture programmes?",
        category=Category.SKILLSFUTURE,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["skillsfuture_employer", "sfec", "etss"],
        answer_keywords=["SFEC", "ETSS", "absentee payroll", "training", "subsidies"],
    ),
    EvalQuestion(
        id="sf_008",
        question="What is the SkillsFuture Work-Study Programme?",
        category=Category.SKILLSFUTURE,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["work_study_programme", "skillsfuture", "fresh_graduate"],
        answer_keywords=["Work-Study", "ITE", "polytechnic", "degree", "on-the-job training"],
    ),
]

# =============================================================================
# Education Questions
# =============================================================================

EDUCATION_QUESTIONS = [
    EvalQuestion(
        id="edu_001",
        question="What is the MOE Financial Assistance Scheme?",
        category=Category.EDUCATION,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.EASY,
        expected_entities=["moe_fas", "financial_assistance", "school_fees"],
        answer_keywords=["MOE FAS", "gross household income", "per capita income", "school fees", "textbooks"],
    ),
    EvalQuestion(
        id="edu_002",
        question="What is the Post-Secondary Education Account (PSEA)?",
        category=Category.EDUCATION,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["psea", "post_secondary", "edusave"],
        answer_keywords=["PSEA", "Edusave", "ITE", "polytechnic", "approved courses"],
    ),
    EvalQuestion(
        id="edu_003",
        question="What bursaries are available for polytechnic students?",
        category=Category.EDUCATION,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["polytechnic_bursary", "moe_bursary", "financial_assistance"],
        answer_keywords=["bursary", "gross household income", "per capita income", "polytechnic"],
    ),
    EvalQuestion(
        id="edu_004",
        question="How does the Tuition Grant Scheme work for university students?",
        category=Category.EDUCATION,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["tuition_grant", "university", "moe"],
        answer_keywords=["Tuition Grant", "subsidised", "service obligation", "Singapore", "3 years"],
    ),
    EvalQuestion(
        id="edu_005",
        question="What financial assistance is available for university students?",
        category=Category.EDUCATION,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["university_financial_aid", "bursary", "study_loan"],
        answer_keywords=["bursary", "Tuition Fee Loan", "Study Loan", "per capita income"],
    ),
    EvalQuestion(
        id="edu_006",
        question="What is the Edusave scheme?",
        category=Category.EDUCATION,
        question_type=QuestionType.SPECIFIC,
        difficulty=Difficulty.EASY,
        expected_entities=["edusave", "edusave_account", "primary_secondary"],
        answer_keywords=["Edusave", "primary", "secondary", "annual contribution", "enrichment"],
    ),
    EvalQuestion(
        id="edu_007",
        question="How can I apply for the MOE Tuition Fee Loan?",
        category=Category.EDUCATION,
        question_type=QuestionType.PROCESS,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["tuition_fee_loan", "moe", "university"],
        answer_keywords=["Tuition Fee Loan", "90%", "interest-free", "repayment", "after graduation"],
    ),
    EvalQuestion(
        id="edu_008",
        question="What scholarships are available for Singapore students?",
        category=Category.EDUCATION,
        question_type=QuestionType.OVERVIEW,
        difficulty=Difficulty.MEDIUM,
        expected_entities=["scholarship", "moe_scholarship", "government_scholarship"],
        answer_keywords=["scholarship", "PSC", "public service", "bond", "merit-based"],
    ),
]

# =============================================================================
# Cross-Category Questions (Multi-hop reasoning)
# =============================================================================

CROSS_CATEGORY_QUESTIONS = [
    EvalQuestion(
        id="cross_001",
        question="I'm a 35-year-old single Singaporean earning $5,000/month. What housing options and grants am I eligible for?",
        category=Category.HDB,
        question_type=QuestionType.MULTI_HOP,
        difficulty=Difficulty.HARD,
        expected_entities=["single_singapore_citizen", "bto", "resale", "enhanced_cpf_housing_grant", "income_ceiling"],
        answer_keywords=["35 years old", "2-room Flexi", "resale", "EHG", "income ceiling"],
    ),
    EvalQuestion(
        id="cross_002",
        question="How do CPF, HDB grants, and income tax interact when buying a first home?",
        category=Category.CPF,
        question_type=QuestionType.MULTI_HOP,
        difficulty=Difficulty.HARD,
        expected_entities=["cpf_oa", "enhanced_cpf_housing_grant", "property_tax", "stamp_duty"],
        answer_keywords=["OA", "EHG", "downpayment", "ABSD", "property tax"],
    ),
    EvalQuestion(
        id="cross_003",
        question="What support is available for seniors who need both healthcare and financial assistance?",
        category=Category.MOH,
        question_type=QuestionType.MULTI_HOP,
        difficulty=Difficulty.HARD,
        expected_entities=["pioneer_generation", "silver_support", "chas", "medisave", "comcare"],
        answer_keywords=["Pioneer", "Merdeka", "Silver Support", "CHAS", "ComCare", "MediSave"],
    ),
    EvalQuestion(
        id="cross_004",
        question="I want to start a business in Singapore. What grants, training subsidies, and regulatory requirements should I know about?",
        category=Category.GRANTS,
        question_type=QuestionType.MULTI_HOP,
        difficulty=Difficulty.HARD,
        expected_entities=["psg", "startup_sg", "skillsfuture_enterprise_credit", "acra", "gobusiness"],
        answer_keywords=["PSG", "Startup SG", "SFEC", "ACRA", "licence", "GoBusiness"],
    ),
    EvalQuestion(
        id="cross_005",
        question="How can a mid-career professional change industries using government support?",
        category=Category.SKILLSFUTURE,
        question_type=QuestionType.MULTI_HOP,
        difficulty=Difficulty.HARD,
        expected_entities=["career_conversion_programme", "skillsfuture_credit", "mid_career_enhanced_subsidy", "wsg"],
        answer_keywords=["CCP", "SkillsFuture Credit", "Mid-Career", "WSG", "training"],
    ),
    EvalQuestion(
        id="cross_006",
        question="What financial support is available for low-income families with school-going children?",
        category=Category.EDUCATION,
        question_type=QuestionType.MULTI_HOP,
        difficulty=Difficulty.HARD,
        expected_entities=["moe_fas", "comcare", "edusave", "school_meals"],
        answer_keywords=["MOE FAS", "ComCare", "Edusave", "textbooks", "school fees", "meals"],
    ),
]


# =============================================================================
# Dataset Assembly
# =============================================================================

def get_all_questions() -> list[EvalQuestion]:
    """Return all evaluation questions."""
    all_questions = []
    all_questions.extend(CPF_QUESTIONS)
    all_questions.extend(HDB_QUESTIONS)
    all_questions.extend(IRAS_QUESTIONS)
    all_questions.extend(MOH_QUESTIONS)
    all_questions.extend(MOM_QUESTIONS)
    all_questions.extend(MSF_QUESTIONS)
    all_questions.extend(GRANTS_QUESTIONS)
    all_questions.extend(SKILLSFUTURE_QUESTIONS)
    all_questions.extend(EDUCATION_QUESTIONS)
    all_questions.extend(CROSS_CATEGORY_QUESTIONS)
    return all_questions


def get_questions_by_category(category: Category) -> list[EvalQuestion]:
    """Return questions for a specific category."""
    all_questions = get_all_questions()
    return [q for q in all_questions if q.category == category]


def get_questions_by_type(question_type: QuestionType) -> list[EvalQuestion]:
    """Return questions of a specific type."""
    all_questions = get_all_questions()
    return [q for q in all_questions if q.question_type == question_type]


def get_questions_by_difficulty(difficulty: Difficulty) -> list[EvalQuestion]:
    """Return questions of a specific difficulty."""
    all_questions = get_all_questions()
    return [q for q in all_questions if q.difficulty == difficulty]


def save_dataset(output_path: Path) -> None:
    """Save the evaluation dataset to JSON."""
    questions = get_all_questions()
    data = {
        "metadata": {
            "version": "1.0",
            "total_questions": len(questions),
            "categories": [c.value for c in Category],
            "question_types": [t.value for t in QuestionType],
            "difficulties": [d.value for d in Difficulty],
        },
        "questions": [q.to_dict() for q in questions],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(questions)} questions to {output_path}")


def load_dataset(input_path: Path) -> list[EvalQuestion]:
    """Load the evaluation dataset from JSON."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for q_data in data["questions"]:
        questions.append(EvalQuestion(
            id=q_data["id"],
            question=q_data["question"],
            category=Category(q_data["category"]),
            question_type=QuestionType(q_data["question_type"]),
            difficulty=Difficulty(q_data["difficulty"]),
            expected_entities=q_data["expected_entities"],
            answer_keywords=q_data["answer_keywords"],
            reference_note=q_data.get("reference_note", ""),
        ))

    return questions


def print_dataset_summary() -> None:
    """Print a summary of the evaluation dataset."""
    questions = get_all_questions()

    print("=" * 60)
    print("GraphRAG Evaluation Dataset Summary")
    print("=" * 60)
    print(f"\nTotal Questions: {len(questions)}")

    print("\nBy Category:")
    for category in Category:
        count = len([q for q in questions if q.category == category])
        print(f"  {category.value}: {count}")

    print("\nBy Question Type:")
    for qtype in QuestionType:
        count = len([q for q in questions if q.question_type == qtype])
        print(f"  {qtype.value}: {count}")

    print("\nBy Difficulty:")
    for diff in Difficulty:
        count = len([q for q in questions if q.difficulty == diff])
        print(f"  {diff.value}: {count}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Print summary
    print_dataset_summary()

    # Save to JSON
    output_path = Path(__file__).parent / "eval_data" / "evaluation_dataset.json"
    save_dataset(output_path)

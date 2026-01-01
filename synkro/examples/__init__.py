"""Built-in example policies for instant demos."""

EXPENSE_POLICY = """# Company Expense Policy

## Approval Thresholds
- Expenses under $50: No approval required
- Expenses $50-$500: Manager approval required
- Expenses over $500: VP approval required

## Receipt Requirements
- All expenses over $25 must have a receipt
- Digital receipts are acceptable
- Missing receipts require written justification within 48 hours

## Categories
- Travel: Flights, hotels, ground transportation, meals while traveling
- Meals: Client meals, team events (max $75/person)
- Software: Must be on pre-approved list, exceptions need IT approval
- Equipment: Must be on asset tracking list if over $200
- Office Supplies: Under $100 can be purchased directly

## Reimbursement Timeline
- Submit expenses within 30 days of purchase
- Reimbursements processed within 14 business days
- Late submissions require manager exception approval
"""

HR_HANDBOOK = """# Employee Handbook

## Work Hours
- Standard work week is 40 hours, Monday through Friday
- Core hours are 10am to 3pm when all employees should be available
- Flexible scheduling allowed with manager approval

## Time Off
- Full-time employees receive 15 days PTO per year
- PTO accrues monthly (1.25 days per month)
- Unused PTO can roll over up to 5 days
- PTO requests must be submitted 2 weeks in advance for 3+ days

## Remote Work
- Hybrid schedule: minimum 2 days in office per week
- Fully remote requires director approval
- Home office stipend of $500 for remote workers

## Performance Reviews
- Annual reviews conducted in December
- Mid-year check-ins in June
- Goals set at start of fiscal year
- Promotions considered during annual review cycle only
"""

REFUND_POLICY = """# Return and Refund Policy

## Eligibility
- Items can be returned within 30 days of purchase
- Items must be unused and in original packaging
- Receipt or proof of purchase required

## Exceptions
- Final sale items cannot be returned
- Personalized items cannot be returned
- Perishable goods cannot be returned after 7 days

## Refund Process
- Refunds issued to original payment method
- Processing takes 5-10 business days
- Shipping costs are non-refundable unless item was defective

## Exchanges
- Exchanges available within 30 days
- Size exchanges free of charge
- Different item exchanges treated as return + new purchase

## Defective Items
- Report defects within 14 days
- Photos required for defect claims
- Replacement or full refund offered for confirmed defects
"""

SUPPORT_GUIDELINES = """# Customer Support Guidelines

## Response Times
- Chat: Respond within 2 minutes
- Email: Respond within 4 hours during business hours
- Phone: Answer within 30 seconds, max hold time 3 minutes

## Escalation Tiers
- Tier 1: General questions, password resets, basic troubleshooting
- Tier 2: Technical issues, billing disputes, account problems
- Tier 3: Complex technical issues, executive escalations

## Refund Authority
- Tier 1 can issue refunds up to $50
- Tier 2 can issue refunds up to $200
- Tier 3 or manager approval needed for refunds over $200

## Documentation
- Log all customer interactions in CRM
- Include customer sentiment and issue category
- Note any promised follow-ups with deadlines
"""

SECURITY_POLICY = """# Information Security Policy

## Password Requirements
- Minimum 12 characters
- Must include uppercase, lowercase, number, and symbol
- Change every 90 days
- Cannot reuse last 10 passwords

## Access Control
- Principle of least privilege applies
- Access requests require manager approval
- Quarterly access reviews mandatory
- Terminate access within 24 hours of employee departure

## Data Classification
- Public: Marketing materials, job postings
- Internal: Company announcements, policies
- Confidential: Customer data, financials
- Restricted: PII, payment info, credentials

## Incident Response
- Report security incidents within 1 hour
- Do not attempt to investigate independently
- Preserve evidence (don't delete logs or files)
- Security team leads all incident response
"""

# All policies available as a list
ALL_POLICIES = [
    ("expense", EXPENSE_POLICY),
    ("hr", HR_HANDBOOK),
    ("refund", REFUND_POLICY),
    ("support", SUPPORT_GUIDELINES),
    ("security", SECURITY_POLICY),
]

__all__ = [
    "EXPENSE_POLICY",
    "HR_HANDBOOK", 
    "REFUND_POLICY",
    "SUPPORT_GUIDELINES",
    "SECURITY_POLICY",
    "ALL_POLICIES",
]


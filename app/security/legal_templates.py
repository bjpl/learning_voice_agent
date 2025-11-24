"""
Legal Document Templates - Plan A Security

GDPR/CCPA compliant privacy policy and terms of service templates.
These are customizable templates that should be reviewed by legal counsel.
"""

from datetime import datetime
from typing import Optional

PRIVACY_POLICY_TEMPLATE = """
# Privacy Policy

**Last Updated:** {last_updated}

## 1. Introduction

{company_name} ("we," "our," or "us") respects your privacy and is committed to protecting your personal data. This privacy policy explains how we collect, use, and safeguard your information when you use our Learning Voice Agent service.

## 2. Information We Collect

### 2.1 Information You Provide
- **Account Information:** Email address, name, and password when you create an account
- **Voice Data:** Audio recordings when you use our voice features
- **Conversation Data:** Text transcripts and conversation history
- **Learning Data:** Topics, preferences, and learning progress

### 2.2 Information Collected Automatically
- **Usage Data:** How you interact with our service
- **Device Information:** Device type, operating system, browser type
- **Log Data:** IP address, access times, pages viewed

## 3. How We Use Your Information

We use your information to:
- Provide and improve our services
- Personalize your learning experience
- Process and respond to your voice inputs
- Send service-related communications
- Ensure security and prevent fraud
- Comply with legal obligations

## 4. Legal Basis for Processing (GDPR)

We process your data based on:
- **Consent:** When you explicitly agree (e.g., voice recording)
- **Contract:** To provide services you requested
- **Legitimate Interest:** To improve our services
- **Legal Obligation:** To comply with applicable laws

## 5. Data Sharing

We do not sell your personal data. We may share data with:
- **Service Providers:** Who help us operate our service (cloud hosting, AI processing)
- **Legal Requirements:** When required by law or to protect rights
- **Business Transfers:** In case of merger or acquisition

## 6. Data Retention

We retain your data for:
- **Account Data:** As long as your account is active, plus {retention_period} after deletion
- **Conversation Data:** {conversation_retention_days} days, then automatically deleted
- **Voice Recordings:** Processed and deleted immediately after transcription (not stored)

## 7. Your Rights

### 7.1 GDPR Rights (EU Residents)
- **Access:** Request a copy of your data
- **Rectification:** Correct inaccurate data
- **Erasure:** Request deletion ("right to be forgotten")
- **Portability:** Receive data in machine-readable format
- **Restriction:** Limit how we process your data
- **Objection:** Object to certain processing

### 7.2 CCPA Rights (California Residents)
- **Know:** What personal information we collect
- **Delete:** Request deletion of your data
- **Opt-Out:** Of sale of personal information (we don't sell data)
- **Non-Discrimination:** Equal service regardless of rights exercised

## 8. Exercising Your Rights

To exercise your data rights:
- **In-App:** Use Settings > Privacy > Data Export/Deletion
- **API:** POST to /api/gdpr/export or /api/gdpr/delete
- **Email:** Contact {privacy_email}

We will respond within 30 days.

## 9. Data Security

We implement security measures including:
- Encryption in transit (TLS 1.3) and at rest
- Access controls and authentication
- Regular security audits
- Employee security training

## 10. International Transfers

Your data may be transferred to servers in {data_locations}. We ensure adequate protection through:
- Standard Contractual Clauses (SCCs)
- Appropriate security measures

## 11. Children's Privacy

Our service is not intended for children under {minimum_age}. We do not knowingly collect data from children.

## 12. Changes to This Policy

We may update this policy. Significant changes will be notified via email or in-app notification.

## 13. Contact Us

**Data Protection Officer:** {dpo_name}
**Email:** {privacy_email}
**Address:** {company_address}

---

For GDPR inquiries, you may also contact your local data protection authority.
"""

TERMS_OF_SERVICE_TEMPLATE = """
# Terms of Service

**Effective Date:** {effective_date}

## 1. Acceptance of Terms

By accessing or using the Learning Voice Agent service ("Service"), you agree to these Terms of Service ("Terms"). If you disagree, do not use the Service.

## 2. Description of Service

The Service provides:
- Voice-based learning capture and conversation
- AI-powered responses and insights
- Learning progress tracking and analytics
- Data export and management tools

## 3. User Accounts

### 3.1 Registration
- You must provide accurate information
- You must be at least {minimum_age} years old
- You are responsible for maintaining account security
- One account per person

### 3.2 Account Security
- Use a strong, unique password
- Enable two-factor authentication when available
- Notify us immediately of unauthorized access
- Do not share your credentials

## 4. Acceptable Use

You agree NOT to:
- Violate any laws or regulations
- Infringe intellectual property rights
- Transmit malware or harmful code
- Attempt to gain unauthorized access
- Use the Service for illegal purposes
- Harass, abuse, or harm others
- Impersonate any person or entity
- Interfere with Service operation

## 5. Intellectual Property

### 5.1 Our Content
- The Service and its content are owned by {company_name}
- You receive a limited, non-exclusive license to use the Service
- You may not copy, modify, or distribute our content

### 5.2 Your Content
- You retain ownership of content you create
- You grant us a license to process your content for Service operation
- You are responsible for content you submit

## 6. AI-Generated Content

- AI responses are generated automatically
- We do not guarantee accuracy or completeness
- AI content should not be relied upon for critical decisions
- You are responsible for verifying important information

## 7. Privacy

Your use of the Service is subject to our Privacy Policy, which describes how we collect, use, and protect your data.

## 8. Fees and Payment

### 8.1 Free Tier
- Basic features available at no cost
- Subject to usage limits

### 8.2 Premium Features
- Additional features may require payment
- Pricing and terms disclosed before purchase
- Refunds subject to our refund policy

## 9. Termination

### 9.1 By You
- You may delete your account at any time
- Data will be deleted per our retention policy

### 9.2 By Us
We may suspend or terminate your account for:
- Violation of these Terms
- Illegal activity
- Extended inactivity
- Non-payment (for paid features)

## 10. Disclaimers

THE SERVICE IS PROVIDED "AS IS" WITHOUT WARRANTIES OF ANY KIND. WE DISCLAIM ALL WARRANTIES, EXPRESS OR IMPLIED, INCLUDING MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.

## 11. Limitation of Liability

TO THE MAXIMUM EXTENT PERMITTED BY LAW, WE SHALL NOT BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES ARISING FROM YOUR USE OF THE SERVICE.

## 12. Indemnification

You agree to indemnify and hold us harmless from claims arising from:
- Your use of the Service
- Your content
- Your violation of these Terms
- Your violation of any rights of another

## 13. Governing Law

These Terms are governed by the laws of {governing_jurisdiction}, without regard to conflict of law principles.

## 14. Dispute Resolution

### 14.1 Informal Resolution
Before filing a claim, you agree to contact us to attempt resolution.

### 14.2 Arbitration
Disputes will be resolved through binding arbitration in {arbitration_location}, except for claims eligible for small claims court.

## 15. Changes to Terms

We may modify these Terms. Continued use after changes constitutes acceptance. Material changes will be notified 30 days in advance.

## 16. General

- **Entire Agreement:** These Terms constitute the entire agreement
- **Severability:** Invalid provisions will not affect other provisions
- **Waiver:** Failure to enforce a right does not waive that right
- **Assignment:** You may not assign these Terms without consent

## 17. Contact

For questions about these Terms:
- **Email:** {legal_email}
- **Address:** {company_address}
"""

COOKIE_POLICY_TEMPLATE = """
# Cookie Policy

**Last Updated:** {last_updated}

## What Are Cookies?

Cookies are small text files stored on your device when you visit websites. They help us remember your preferences and improve your experience.

## Types of Cookies We Use

### Essential Cookies
Required for the Service to function. Cannot be disabled.
- Session management
- Authentication
- Security features

### Functional Cookies
Remember your preferences.
- Language settings
- Theme preferences
- Display options

### Analytics Cookies
Help us understand how you use the Service.
- Usage patterns
- Feature popularity
- Error tracking

## Managing Cookies

You can control cookies through:
- **Browser Settings:** Block or delete cookies
- **In-App Settings:** Toggle optional cookies
- **Do Not Track:** We honor DNT signals

## Third-Party Cookies

We may use third-party services that set their own cookies:
- Analytics providers
- Authentication services
- CDN providers

## Updates

We may update this policy. Check the "Last Updated" date for changes.

## Contact

Questions? Contact us at {privacy_email}
"""


class LegalDocumentGenerator:
    """Generate legal documents from templates."""

    DEFAULT_CONFIG = {
        "company_name": "Learning Voice Agent",
        "last_updated": datetime.now().strftime("%B %d, %Y"),
        "effective_date": datetime.now().strftime("%B %d, %Y"),
        "privacy_email": "privacy@example.com",
        "legal_email": "legal@example.com",
        "dpo_name": "Data Protection Officer",
        "company_address": "[Company Address]",
        "minimum_age": "13",
        "retention_period": "30 days",
        "conversation_retention_days": "90",
        "data_locations": "United States",
        "governing_jurisdiction": "[Jurisdiction]",
        "arbitration_location": "[Location]",
    }

    @classmethod
    def generate_privacy_policy(cls, config: Optional[dict] = None) -> str:
        """Generate privacy policy with custom config."""
        merged_config = {**cls.DEFAULT_CONFIG, **(config or {})}
        return PRIVACY_POLICY_TEMPLATE.format(**merged_config)

    @classmethod
    def generate_terms_of_service(cls, config: Optional[dict] = None) -> str:
        """Generate terms of service with custom config."""
        merged_config = {**cls.DEFAULT_CONFIG, **(config or {})}
        return TERMS_OF_SERVICE_TEMPLATE.format(**merged_config)

    @classmethod
    def generate_cookie_policy(cls, config: Optional[dict] = None) -> str:
        """Generate cookie policy with custom config."""
        merged_config = {**cls.DEFAULT_CONFIG, **(config or {})}
        return COOKIE_POLICY_TEMPLATE.format(**merged_config)

    @classmethod
    def get_all_documents(cls, config: Optional[dict] = None) -> dict:
        """Get all legal documents."""
        return {
            "privacy_policy": cls.generate_privacy_policy(config),
            "terms_of_service": cls.generate_terms_of_service(config),
            "cookie_policy": cls.generate_cookie_policy(config),
        }

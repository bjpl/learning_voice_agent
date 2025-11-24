"""
Legal Document Routes - Privacy Policy, Terms of Service, etc.

These endpoints serve the legal documents required for compliance.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, PlainTextResponse

from app.security.legal_templates import LegalDocumentGenerator

legal_router = APIRouter(prefix="/legal", tags=["legal"])


@legal_router.get("/privacy", response_class=PlainTextResponse)
async def get_privacy_policy():
    """
    Get the Privacy Policy.

    Returns the privacy policy in Markdown format.
    """
    return LegalDocumentGenerator.generate_privacy_policy()


@legal_router.get("/terms", response_class=PlainTextResponse)
async def get_terms_of_service():
    """
    Get the Terms of Service.

    Returns the terms of service in Markdown format.
    """
    return LegalDocumentGenerator.generate_terms_of_service()


@legal_router.get("/cookies", response_class=PlainTextResponse)
async def get_cookie_policy():
    """
    Get the Cookie Policy.

    Returns the cookie policy in Markdown format.
    """
    return LegalDocumentGenerator.generate_cookie_policy()


@legal_router.get("/all")
async def get_all_legal_documents():
    """
    Get all legal documents.

    Returns all legal documents in a JSON response.
    """
    return LegalDocumentGenerator.get_all_documents()


def setup_legal_routes(app):
    """Setup legal routes on the FastAPI app."""
    app.include_router(legal_router)

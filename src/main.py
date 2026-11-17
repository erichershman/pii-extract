"""ASGI entry point: ``uvicorn main:app --app-dir src``.

The application itself (routes and the request/response models, including the
``labels`` example ``["person", "email", "phone number", "ssn"]``) lives in
``pii_extract.api`` so it ships with the installed package. This module only
instantiates it from the environment.
"""

from pii_extract.api import create_app

app = create_app()

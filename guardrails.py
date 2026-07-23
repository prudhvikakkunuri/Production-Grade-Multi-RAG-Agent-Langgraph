import re
# ============================================================
# VAULTIFY AI - GUARDRAILS LAYER
# ============================================================
# ============================================================
# CONFIGURATION
# ============================================================

MAX_INPUT_LENGTH = 2000


# ============================================================
# PROMPT INJECTION / JAILBREAK PATTERNS
# ============================================================

BLOCKED_PATTERNS = [
    # --------------------------------------------------------
    # Prompt Injection
    # --------------------------------------------------------
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?instructions",
    r"forget\s+(all\s+)?previous\s+instructions",
    r"override\s+(your\s+)?instructions",
    r"bypass\s+(your\s+)?rules",
    r"disregard\s+(all\s+)?previous\s+instructions",
    # --------------------------------------------------------
    # System Prompt Extraction
    # --------------------------------------------------------
    r"reveal\s+(your\s+)?system\s+prompt",
    r"show\s+(me\s+)?your\s+system\s+prompt",
    r"what\s+is\s+your\s+system\s+prompt",
    r"print\s+(your\s+)?system\s+prompt",
    r"display\s+(your\s+)?system\s+prompt",
    r"show\s+(me\s+)?hidden\s+instructions",
    r"reveal\s+(your\s+)?hidden\s+instructions",
    r"developer\s+instructions",
    # --------------------------------------------------------
    # Jailbreak Attempts
    # --------------------------------------------------------
    r"\bjailbreak\b",
    r"\bDAN\s+mode\b",
    r"act\s+as\s+an?\s+unrestricted",
    r"pretend\s+you\s+have\s+no\s+restrictions",
    r"pretend\s+you\s+have\s+no\s+rules",
    # --------------------------------------------------------
    # API Key / Credential Extraction
    # --------------------------------------------------------
    r"show\s+(me\s+)?(?:the\s+)?api\s*key",
    r"reveal\s+(?:the\s+)?api\s*key",
    r"give\s+(me\s+)?(?:the\s+)?api\s*key",
    r"print\s+(?:the\s+)?api\s*key",
    r"show\s+(me\s+)?(?:the\s+)?password",
    r"reveal\s+(?:the\s+)?password",
    r"give\s+(me\s+)?(?:the\s+)?password",
    r"show\s+(me\s+)?(?:the\s+)?secret\s*key",
    r"reveal\s+(?:the\s+)?secret\s*key",

    # --------------------------------------------------------
    # Environment Variables
    # --------------------------------------------------------
    r"show\s+(me\s+)?(?:the\s+)?environment\s+variables",
    r"print\s+(?:the\s+)?environment\s+variables",
    r"reveal\s+(?:the\s+)?environment\s+variables",
    r"show\s+(me\s+)?(?:the\s+)?\.env",
    r"print\s+(?:the\s+)?\.env",
    r"reveal\s+(?:the\s+)?\.env",

    # --------------------------------------------------------
    # Database Extraction
    # --------------------------------------------------------
    r"dump\s+(?:the\s+)?database",
    r"show\s+(me\s+)?(?:the\s+)?database\s+credentials",
    r"reveal\s+(?:the\s+)?database\s+credentials",
    r"give\s+(me\s+)?(?:the\s+)?database\s+credentials",
]

# ============================================================
# SENSITIVE DATA PATTERNS
# ============================================================
SENSITIVE_PATTERNS = {
    # --------------------------------------------------------
    # Credit / Debit Card
    # Supports spaces and hyphens
    # Example:
    # 4532 0151 1283 0366
    # --------------------------------------------------------
    "credit_card":
        r"\b(?:\d[ -]*?){13,19}\b",
    # --------------------------------------------------------
    # CVV / CVC
    # Example:
    # CVV: 123
    # CVC = 456
    # --------------------------------------------------------
    "cvv":
        r"\b(?:cvv|cvc|security\s+code)"
        r"\s*[:=\-]?\s*\d{3,4}\b",
    # --------------------------------------------------------
    # Password
    # Example:
    # password = Vaultify123
    # pwd: hello123
    # --------------------------------------------------------
    "password":
        r"\b(?:password|passwd|pwd)"
        r"\s*[:=]\s*\S+",
    # --------------------------------------------------------
    # API Key / Secret Key
    # Example:
    # API_KEY = abc123xyz
    # secret_key: xyz123
    # --------------------------------------------------------
    "api_key":
        r"\b(?:api[_\s-]?key|secret[_\s-]?key)"
        r"\s*[:=]\s*\S+",
    # --------------------------------------------------------
    # OTP
    # Example:
    # OTP: 123456
    # OTP = 987654
    # --------------------------------------------------------
    "otp":
        r"\b(?:otp|one[\s-]?time[\s-]?password)"
        r"\s*[:=\-]?\s*\d{4,8}\b",
}
# ============================================================
# REDACTION LABELS
# ============================================================
REDACTION_LABELS = {
    "credit_card": "[REDACTED CARD NUMBER]",
    "cvv": "[REDACTED CVV]",
    "password": "[REDACTED PASSWORD]",
    "api_key": "[REDACTED API KEY]",
    "otp": "[REDACTED OTP]",
}
# ============================================================
# DETECT PROMPT INJECTION
# ============================================================
def detect_prompt_injection(user_input: str) -> bool:
    """
    Detects common prompt injection, jailbreak,
    system prompt extraction and credential
    extraction attempts.
    """
    if not user_input:
        return False
    for pattern in BLOCKED_PATTERNS:
        if re.search(
            pattern,
            user_input,
            re.IGNORECASE
        ):
            return True
    return False

# ============================================================
# DETECT SENSITIVE INFORMATION
# ============================================================

def detect_sensitive_data(user_input: str) -> list:
    """
    Detects sensitive information contained
    inside the user's query.

    Returns a list containing the detected
    sensitive data categories.
    """
    detected = []
    if not user_input:
        return detected
    for data_type, pattern in SENSITIVE_PATTERNS.items():
        if re.search(
            pattern,
            user_input,
            re.IGNORECASE
        ):
            detected.append(data_type)
    return detected


# ============================================================
# REDACT SENSITIVE INFORMATION
# ============================================================

def redact_sensitive_data(text: str) -> str:
    """
    Replaces sensitive information with
    safe redaction labels before the query
    is passed to the RAG pipeline.
    """

    if not text:
        return text
    redacted_text = text

    for data_type, pattern in SENSITIVE_PATTERNS.items():

        replacement = REDACTION_LABELS.get(
            data_type,
            "[REDACTED SENSITIVE DATA]"
        )
        redacted_text = re.sub(
            pattern,
            replacement,
            redacted_text,
            flags=re.IGNORECASE
        )

    return redacted_text

# ============================================================
# INPUT GUARDRAIL
# ============================================================

def apply_guardrails(user_input: str) -> dict:
    """
    Main input guardrail.
    Performs:
    1. Empty input validation
    2. Input length validation
    3. Prompt injection detection
    4. Jailbreak detection
    5. System prompt protection
    6. Sensitive data detection
    7. Sensitive data redaction

    Returns a dictionary containing:

    allowed
    message
    sanitized_input
    redacted
    detected_sensitive_data
    """
    # --------------------------------------------------------
    # 1. Empty Input
    # --------------------------------------------------------

    if not user_input or not user_input.strip():
        return {
            "allowed": False,
            "message":"Please enter a valid question.",
            "sanitized_input": None,
            "redacted": False,
            "detected_sensitive_data": []
        }

    # --------------------------------------------------------
    # 2. Input Length
    # --------------------------------------------------------

    if len(user_input) > MAX_INPUT_LENGTH:
        return {
            "allowed": False,
            "message": (
                f"Your message is too long. "
                f"Please keep it under "
                f"{MAX_INPUT_LENGTH} characters."
            ),
            "sanitized_input": None,
            "redacted": False,
            "detected_sensitive_data": []
        }

    # --------------------------------------------------------
    # 3. Prompt Injection / Jailbreak Detection
    # --------------------------------------------------------
    if detect_prompt_injection(user_input):
        return {
            "allowed": False,
            "message": (
                "🔒 I can't process requests that attempt "
                "to modify, bypass, or reveal my internal "
                "instructions or confidential information."
            ),
            "sanitized_input": None,
            "redacted": False,
            "detected_sensitive_data": []
        }


    # --------------------------------------------------------
    # 4. Detect Sensitive Information
    # --------------------------------------------------------

    sensitive_data = detect_sensitive_data(
        user_input
    )
    # --------------------------------------------------------
    # 5. Redact Sensitive Information
    # --------------------------------------------------------
    if sensitive_data:
        redacted_input = redact_sensitive_data(
            user_input
        )
        return {
            # We still allow the query
            "allowed": True,
            "message": None,
            # This is what should be sent to RAG
            "sanitized_input": redacted_input,
            "redacted": True,
            "detected_sensitive_data":
                sensitive_data
        }

    # --------------------------------------------------------
    # 6. Safe Input
    # --------------------------------------------------------
    return {
        "allowed": True,
        "message": None,
        "sanitized_input": user_input,
        "redacted": False,
        "detected_sensitive_data": []
    }

# ============================================================
# OUTPUT GUARDRAIL
# ============================================================

def validate_output(response: str) -> str:

    """
    Checks the LLM/RAG response for possible
    exposed credentials or secrets.

    If something suspicious is detected,
    the response will not be displayed.
    """

    if not response:
        return (
            "I couldn't generate a response. "
            "Please try again."
        )

    # --------------------------------------------------------
    # Secret patterns that should never appear
    # in the final response
    # --------------------------------------------------------

    secret_patterns = [
        # OpenAI-style keys
        r"sk-[A-Za-z0-9_-]{20,}",

        # Google API keys
        r"AIza[A-Za-z0-9_-]{20,}",

        # Generic API / secret key
        r"(?:api[_\s-]?key|secret[_\s-]?key)"
        r"\s*[:=]\s*[\"']?"
        r"[A-Za-z0-9_\-./]{8,}",

        # Password exposure
        r"(?:password|passwd|pwd)"
        r"\s*[:=]\s*[\"']?"
        r"\S{6,}",
    ]

    for pattern in secret_patterns:

        if re.search(
            pattern,
            response,
            re.IGNORECASE
        ):
            return (
                "🔒 I can't display this response because "
                "it may contain sensitive or confidential "
                "information."
            )


    return response
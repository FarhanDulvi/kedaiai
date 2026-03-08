"""
KedaiAI — Multilingual AI Business Assistant for Malaysian SMEs
GDG Cloud KL: Build with AI Gemini Hackathon

Core Features:
1. Multimodal Input  — Accepts text OR screenshot of customer messages
2. Language Detection — Identifies BM, EN, Manglish, Mandarin, Tamil, mixed
3. Intent Analysis    — Understands orders, inquiries, complaints
4. Multilingual Reply — Generates replies in BM, EN, and Mandarin
5. Order Extraction   — Structures order data into JSON
"""

import os
import json
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify

import google.generativeai as genai
from PIL import Image

# ─────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────
app = Flask(__name__)

# Configure Gemini API — set your key in environment variable or replace below
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAP6FzT0Izm84zkjcdZhYZFaSIG9h8tTg0")
genai.configure(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────
# System Prompt — The Heart of KedaiAI
# This is where we steer Gemini's behavior with
# deep Malaysian cultural context
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are KedaiAI, an intelligent multilingual business assistant built specifically for Malaysian micro-enterprises and SMEs.

YOUR ROLE:
You help Malaysian small business owners (hawker stalls, kedai runcit, online sellers, home bakers, etc.) instantly understand and respond to customer messages in any language.

YOUR CAPABILITIES:
1. Language Detection — Identify whether the customer writes in Bahasa Melayu, English, Manglish, Mandarin Chinese, Tamil, or mixed code-switching.
2. Intent Analysis — Classify the customer's intent: ordering, price inquiry, delivery request, complaint, feedback, reservation, or general inquiry.
3. Sentiment Analysis — Gauge the customer's mood to adjust response tone.
4. Order Extraction — If the message contains an order, extract each item with quantity and special instructions.
5. Multilingual Reply Generation — Draft warm, professional replies in BM, EN, and Mandarin that the business owner can copy-paste to reply.

DEEP MALAYSIAN CULTURAL CONTEXT:
- Malaysians frequently code-switch mid-sentence (e.g., "Boss, nak order 2 nasi lemak, extra sambal please, how much ah?")
- "Manglish" uses particles: "lah", "mah", "lor", "kan", "leh", "geh"
- Common terms: "tapau/dabao" (takeaway), "boss" (generic address), "kakak/abang" (older sister/brother), "makcik/pakcik" (aunt/uncle)
- Malaysian food terms: nasi lemak, roti canai, teh tarik, kopi-o, mee goreng, char kuey teow, cendol, apam balik, etc.
- Currency: Ringgit Malaysia (RM). Typical hawker prices: RM 5-15 per dish.
- Business hours vary: hawker stalls often open early morning (6 AM) or operate late night till midnight.
- Respectful greetings matter: "Assalamualaikum" (BM/Muslim), "Hi/Hello" (English), "你好" (Mandarin).
- Malaysian sellers often use WhatsApp for business orders.
- During Ramadan, many sellers offer special buka puasa (breaking fast) menus.

RESPONSE FORMAT — You MUST return valid JSON with this exact structure:
{
  "detected_language": "Manglish | Bahasa Melayu | English | Mandarin | Tamil | Mixed",
  "confidence": 0.95,
  "customer_intent": "Food Order | Price Inquiry | Delivery Request | Complaint | Feedback | Reservation | General Inquiry",
  "sentiment": "positive | neutral | negative",
  "intent_summary": "Brief 1-2 sentence summary of what the customer wants",
  "order_items": [
    {
      "item": "Item name",
      "quantity": 1,
      "special_instructions": "Any modifications or null"
    }
  ],
  "estimated_total": "RM X.XX or null if cannot estimate",
  "reply_bm": "Professional and warm reply in Bahasa Melayu",
  "reply_en": "Friendly and clear reply in English",
  "reply_mandarin": "Polite and helpful reply in Mandarin Chinese (use Chinese characters)",
  "reply_original_language": "Reply matching the customer's detected language and style — if they use Manglish, reply in Manglish with appropriate particles",
  "business_tip": "One actionable tip for the business owner about this interaction"
}

CRITICAL RULES:
- If analyzing a screenshot/image, first READ all text from the image, then analyze it as a customer message.
- Match the formality and warmth level of the customer. Manglish customer = Manglish reply.
- For food orders, always confirm items and ask about delivery/pickup.
- Include culturally appropriate greetings and closings in each language.
- If order items are detected, suggest a polite way to confirm the order.
- If no order items, return an empty array for order_items and null for estimated_total.
- ALWAYS return valid JSON. No markdown, no code blocks, just pure JSON.
"""

# ─────────────────────────────────────────────
# Initialize Gemini Model
# Using gemini-2.0-flash for speed (critical for demos)
# JSON response mode for structured output
# ─────────────────────────────────────────────
model = genai.GenerativeModel(
    model_name="gemini-3-flash-preview",
    system_instruction=SYSTEM_PROMPT,
    generation_config={
        "response_mime_type": "application/json",
        "temperature": 0.7,
        "max_output_tokens": 2048,
    },
)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main application page."""
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Main analysis endpoint.
    Accepts JSON with:
      - text: string (customer message)
      - image: string (base64 encoded image, optional)
    Returns structured Gemini analysis.
    """
    try:
        data = request.json
        text_input = data.get("text", "").strip()
        image_data = data.get("image", None)

        # Validate: need at least text or image
        if not text_input and not image_data:
            return jsonify({"success": False, "error": "Please provide a message or upload a screenshot."}), 400

        # Build the content array for Gemini
        contents = []

        # Handle image input (multimodal feature)
        if image_data:
            try:
                # Remove data URL prefix if present
                if "," in image_data:
                    image_bytes = base64.b64decode(image_data.split(",")[1])
                else:
                    image_bytes = base64.b64decode(image_data)

                image = Image.open(BytesIO(image_bytes))
                contents.append(image)

                if text_input:
                    contents.append(
                        f"I've uploaded a screenshot of a customer message. "
                        f"Please read the text from the image and analyze it. "
                        f"Additional context: {text_input}"
                    )
                else:
                    contents.append(
                        "I've uploaded a screenshot of a customer message (e.g., from WhatsApp). "
                        "Please read all text from the image and analyze it as a customer message."
                    )
            except Exception as img_err:
                return jsonify({"success": False, "error": f"Image processing failed: {str(img_err)}"}), 400

        else:
            # Text-only analysis
            contents.append(
                f"Analyze this customer message and provide the structured analysis:\n\n"
                f'"{text_input}"'
            )

        # Call Gemini API
        response = model.generate_content(contents)

        # Parse JSON response
        result = json.loads(response.text)

        return jsonify({"success": True, "data": result})

    except json.JSONDecodeError as je:
        # Gemini sometimes returns imperfect JSON — try to salvage
        return jsonify({
            "success": False,
            "error": "AI response parsing failed. Please try again.",
            "raw_response": response.text if "response" in dir() else None,
        }), 500

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint for Cloud Run."""
    return jsonify({"status": "healthy", "service": "KedaiAI"})


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)

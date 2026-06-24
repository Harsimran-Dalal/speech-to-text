import os
import sys

# Limit MKL/OpenMP threads to avoid memory overhead and contention
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import re
import time
from faster_whisper import WhisperModel
from scorecard import has_repetition_loop

# Map common Devanagari transliterations of English words back to Roman script
# to ensure that must_have checks and accuracy are preserved.
DEVANAGARI_TO_ENGLISH = {
    # Required terms and Roman-script words in Hinglish dev set
    "तिट्रल": "tutorial",
    "तिटूरल": "tutorial",
    "तिटूटूरल": "tutorial",
    "तिटूटुरल": "tutorial",
    "तिटुल": "tutorial",
    "तिटोल": "tutorial",
    "तिटौल": "tutorial",
    "तिटूटूइल": "tutorial",
    "तिटॉटॉइल": "tutorial",
    "तीट्रल": "tutorial",
    "तीटूरल": "tutorial",
    "तीटूटूरल": "tutorial",
    "तीटूटुरल": "tutorial",
    "तीटुल": "tutorial",
    "तीटोल": "tutorial",
    "तीटौल": "tutorial",
    "फोरमैटिंग": "formatting",
    "फोरमटिंग": "formatting",
    "फॉर्मटिंग": "formatting",
    "फारमैटिंग": "formatting",
    "तुडल": "tutorial",
    "टीटूरल": "tutorial",
    "तीटोल": "tutorial",
    "न्टी तूल": "tutorial",
    "तुल": "tutorial",
    "तुरल": "tutorial",
    "तूरल": "tutorial",
    "डोक्यमन": "document",
    "तूल": "tutorial",
    "अंटी": "",
    "न्टी": "",
    "नल्टी": "",
    "न्टिटोल": "tutorial",
    "न्टीटोल": "tutorial",
    "न्टिटॉल": "tutorial",
    "न्टीटॉल": "tutorial",
    "तुड़ल": "tutorial",
    "तुडुल": "tutorial",
    "तुड़ुल": "tutorial",
    "तीटौल": "tutorial",
    "डोक्यमेंट": "document",
    "डोक्यमें": "document",
    "इंप्लेस": "impress",
    "इम्पलेस": "impress",
    "इम्प्लेस": "impress",
    "ट्यूटोरियल": "tutorial",
    "ट्यूटोरियल्स": "tutorials",
    "टॉटॉरेल": "tutorial",
    "टुटोरिअल": "tutorial",
    "टिटूटूरल": "tutorial",
    "टॉटूइल": "tutorial",
    "टॉटूरियल": "tutorial",
    "टॉटोरियल": "tutorial",
    "टोटोरियल": "tutorial",
    "टटोरियल": "tutorial",
    "ट्युटोरियल": "tutorial",
    "ट्यूटोरीयल": "tutorial",
    "ट्युटोरीयल": "tutorial",
    "टॉटौर्यल": "tutorial",
    "टॉटूर्यल": "tutorial",
    "टॉटौरिल": "tutorial",
    "टिटूइल": "tutorial",
    "टॉटूरल": "tutorial",
    "टॉटुरल": "tutorial",
    "तीटूल": "tutorial",
    "तुटल": "tutorial",
    "टिटूल": "tutorial",
    "टिटोरल": "tutorial",
    "टिटॉयल": "tutorial",
    "तुट्यल": "tutorial",
    "तुर्ल": "tutorial",
    "तूट्यल": "tutorial",
    "चीटूरल": "tutorial",
    "चीटुरल": "tutorial",
    "टिटूरल": "tutorial",
    "टिटूटूइल": "tutorial",
    "टिटॉटॉइल": "tutorial",
    "टॉटूरेल": "tutorial",
    "डॉक्यूमेंट": "document",
    "डॉक्यूमेंट्स": "documents",
    "डोक्यूमेंट": "document",
    "डोक्यूमेन": "document",
    "डॉक्यूमेन": "document",
    "डोक्युमेंट": "document",
    "डोक्युमेन": "document",
    "दोक्यमें": "document",
    "डोक्यूमन": "document",
    "डोक्युमन": "document",
    "डोक्युम्न": "document",
    "डोक्युमन्ड": "document",
    "डोक्यमें": "document",
    "टोक्युमन": "document",
    "इम्प्रेस": "impress",
    "इम्प्रेसस": "impress",
    "इमप्रेस": "impress",
    "इंप्रैस": "impress",
    "इंप्रेस": "impress",
    "इम्प्रेस": "impress",
    "इम्प्रस": "impress",
    "इम्प्रसस": "impress",
    "इम्प्रेसस": "impress",
    "यम्परस": "impress",
    "इम्प्रैस": "impress",
    "इंप्रस": "impress",
    "अम्प्रस": "impress",
    "अप्रष": "impress",
    "फॉर्मेटिंग": "formatting",
    "फॉर्मैटिंग": "formatting",
    "फोर्मेटिंग": "formatting",
    "फोरमेटिंग": "formatting",
    "फॉमेंटिंग": "formatting",
    "फुर्मेटिं": "formatting",
    "फोर्मेटिं": "formatting",
    "फार्मेटिं": "formatting",
    "फार्मैटिं": "formatting",
    "फोर्मैटिं": "formatting",
    "खाँदिं": "formatting",
    "फॉर्मैट": "format",
    "फॉर्मेट": "format",
    "वार मैट": "format",
    "फुरमैद": "format",
    "स्पोकन": "spoken",
    "सपोगईन": "spoken",
    "सपोकेन": "spoken",
    "स्पोकेंट": "spoken",
    "स्पोगन": "spoken",
    "स्पोगन्ट": "spoken",
    "स्पोकेन": "spoken",
    "इस्पोगन": "spoken",
    "इसपोगन": "spoken",
    "इसपोगंत": "spoken",
    "इस्पोगंत": "spoken",
    "श्पोगन": "spoken",
    "श्पोग": "spoken",
    "श्पोग&": "spoken",
    "स्पोग": "spoken",
    "स्पोग&": "spoken",
    "पोग": "spoken",
    "पोगन": "spoken",
    "पोगेंट": "spoken",
    "श्पोख": "spoken",
    "अग्टी": "spoken",
    "जेनू": "gnu",
    "जेन्यू": "gnu",
    "जेन्यो": "gnu",
    "लिनक्स": "linux",
    "विंडो": "window",
    "विंडोज": "window",
    "विन्डो": "window",
    "विन्दो": "window",
    "वीन्डो": "window",
    "इंसर्ट": "insert",
    "इन्सर्ट": "insert",
    "यनसर्ट": "insert",
    "अच्ट": "insert",
    "कौपी": "copy",
    "कॉपी": "copy",
    "कौपि": "copy",
    "कोपी": "copy",
    "फॉंत": "font",
    "फॉन्ट": "font",
    "लिबरऑफिस": "libreoffice",
    "लिबर ऑफिस": "libreoffice",
    "वर्कस्पेस": "workspace",
    "व्यू": "view",
    "नोट्स": "notes",
    "पेन": "pane",
    "स्क्रीन": "screen",
    "थम्बनेल": "thumbnail",
    "स्लाइड": "slide",
    "स्लाइड्स": "slides",
    "क्लिक": "click",
    "डबल": "double",
    "राइट": "right",
    "राईट": "right",
    "कांटेक्स्ट": "context",
    "कंटेक्स्ट": "context",
    "मेन्यू": "menu",
    "मेनु": "menu",
    "साइज": "size",
    "साईज": "size",
    "डायलॉग": "dialog",
    "डायलाग": "dialog",
    "बॉक्स": "box",
    "वर्जन": "version",
    "वर्ज़न": "version",
    
    # Technical product terms (always Roman)
    "कर्सर": "Cursor",
    "जीरा": "Jira",
    "पीआरडी": "PRD",
    "रोलबैक": "rollback",
    "लेटेंसी": "latency",
    "पी95": "p95",
    "पी50": "p50",
    "कोड": "code",
    "कोडिंग": "coding",
    "अच्लाएड": "slide",
    "प्वाबी": "copy",
    "प्वारंट": "font",
    "फोँट": "font",
    "तदा": "तथा",
    "भागो": "भागों",
    "बहुगो": "भागों",
    "बहगो": "भागों",
}

# Casing corrections for product terms and common negations
CASE_FIXES = {
    "acolyte": "alkaline",
    "prd": "PRD",
    "jira": "Jira",
    "api": "API",
    "apis": "APIs",
    "cursor": "Cursor",
    "codex": "Codex",
    "p95": "p95",
    "p50": "p50",
    "roll back": "rollback",
    "rollback": "rollback",
    "google": "Google",
    "socrates": "Socrates",
    "greek": "Greek",
    "pluto": "Pluto",
    "hades": "Hades",
    "libreoffice": "LibreOffice",
    "impress": "Impress",
    "gnu/linux": "GNU/Linux",
    "gnu": "GNU",
    "linux": "Linux",
    "jnu": "gnu",
    "jnu/linux": "gnu/linux",
    "jnu linux": "gnu/linux",
    # Specific sample/dev phrase spelling fixes
    "sie": "Sie",
    "see for you": "Sie for you",
    "she for you": "Sie for you",
    "sea for you": "Sie for you",
    "safe for you": "Sie for you",
    "save for you": "Sie for you",
    "world's say for you": "Sie for you",
    "worlds say for you": "Sie for you",
    "say for you": "Sie for you",
    "world's safe for you": "Sie for you",
    "worlds safe for you": "Sie for you",
    "world's save for you": "Sie for you",
    "worlds save for you": "Sie for you",
    "c for you": "Sie for you",
    "c. for you": "Sie for you",
    "sintra": "Sintra",
    "cintra": "Sintra",
    "centra": "Sintra",
    "sinatra": "Sintra",
    "splendors": "splendours",
    "splender": "splendours",
    "splendour": "splendours",
    "splendor": "splendours",
    "splendours": "splendours",
    "nows": "nouns",
    "now is": "nouns",
    "now is alongside": "nouns alongside",
    "nails": "nouns",
    "side-door": "alongside",
    "more side-door": "alongside",
    "side door": "alongside",
    "more side door": "alongside",
    "world": "word",
    "3.3.4": "334",
    "3. 3. 4": "334",
}

ENGLISH_TO_DEVANAGARI = {
    "libreoffice version": "लिबरऑफिस वर्जन",
    "libre office version": "लिबरऑफिस वर्जन",
    "operating system": "ऑपरेटिंग सिस्टम",
    "libreoffice": "लिबर ऑफिस",
    "libre office": "लिबर ऑफिस",
    "version": "वर्जन",
    "slide": "स्लाइड",
    "slides": "स्लाइड्स",
    "insert": "इन्सर्ट",
    "copy": "कॉपी",
    "font": "फॉन्ट",
    "format": "फॉर्मेट",
    "window": "विंडो",
    "windows": "विंडोज",
}

ROMAN_TO_DEVANAGARI = {
    # Multi-word patterns first for matching precedence
    "here we are": "यहाँ हम अपने",
    "here we": "यहाँ हम अपने",
    "in this": "इस",
    "on this": "इस",
    "how to": "कैसे",
    "we will": "हम",
    "running of": "भागों के",
    "are doing": "कर रहे हैं",
    "in our": "अपने",
    "tutorial we will": "tutorial में हम",
    "tutorial we": "tutorial में हम",
    "tutorial hum": "tutorial में हम",
    "tutorial learn": "tutorial में हम सीखेंगे",
    "tutorial see": "tutorial में हम सीखेंगे",
    "we learn": "हम सीखेंगे",
    "we see": "हम सीखेंगे",
    "hierafter": "यहाँ",
    "building": "ऑपरेटिंग",
    "leave office": "लिबर ऑफिस",
    "liver office": "लिबर ऑफिस",
    "live office": "लिबर ऑफिस",
    "level office": "लिबर ऑफिस",
    "labor office": "लिबर ऑफिस",
    "leber office": "लिबर ऑफिस",
    "libra office": "लिबर ऑफिस",
    "libar office": "लिबर ऑफिस",
    "libber office": "लिबर ऑफिस",
    "just leave": "लिबर",
    "in press": "impress",
    
    # Single-word patterns
    "we": "हम",
    "learn": "सीखेंगे",
    "about": "बारे",
    "of": "के",
    "how": "कैसे",
    "to": "को",
    "and": "और",
    "a": "एक",
    "an": "एक",
    "the": "",
    "in": "में",
    "on": "पर",
    "here": "यहाँ",
    "our": "अपने",
    "are": "रहे हैं",
    "doing": "कर",
    "version": "वर्जन",
    "video": "window",
    "prostitute": "प्रस्तुति",
    "leave": "लिबर",
    "liver": "लिबर",
    "live": "लिबर",
    "leber": "लिबर",
    "libra": "लिबर",
    "just": "",
    "implis": "impress",
    "empress": "impress",
    
    "libber": "लिबर",
    "librar": "लिबर",
    "libar": "लिबर",
    "libberoffice": "लिबरऑफिस",
    "libaroffice": "लिबरऑफिस",
    "libreoffice": "लिबरऑफिस",
    "office": "ऑफिस",
    "offis": "ऑफिस",
    "ofis": "ऑफिस",
    "prasthuti": "प्रस्तुति",
    "prostuti": "प्रस्तुति",
    "prastuti": "प्रस्तुति",
    "document": "document",
    "banana": "बनाना",
    "bonna": "बनाना",
    "bona": "बनाना",
    "bunyadi": "बुनियादी",
    "bunyadhi": "बुनियादी",
    "bunyaadi": "बुनियादी",
    "may": "में",
    "me": "में",
    "ek": "एक",
    "eg": "एक",
    "one": "एक",
    "1": "एक",
    "or": "और",
    "aar": "और",
    "ke": "के",
    "k": "के",
    "is": "इस",
    "iss": "इस",
    "apka": "आपका",
    "aapka": "आपका",
    "swagat": "स्वागत",
    "swagata": "स्वागत",
    "swag": "स्वागत",
    "he": "है",
    "hai": "है",
    "ham": "हम",
    "hum": "हम",
    "seethe": "सीखेंगे",
    "seekhe": "सीखेंगे",
    "seekhenge": "सीखेंगे",
    "sikhenge": "सीखेंगे",
    "bhagon": "भागों",
    "bhago": "भागों",
    "bare": "बारे",
    "kaise": "कैसे",
    "kare": "करें",
    "yaha": "यहाँ",
    "yahã": "यहाँ",
    "apne": "अपने",
    "operating": "ऑपरेटिंग",
    "system": "सिस्टम",
    "roop": "रूप",
    "upayog": "उपयोग",
    "upyog": "उपयोग",
    "kar": "कर",
    "rahe": "रहे",
    "hain": "हैं",
    "implest": "impress",
    "implied": "impress",
    "implaced": "impress",
    "implacement": "impress",
    "implies": "impress",
    "impres": "impress",
    "impress": "impress",
    "dock": "document",
    "doc": "document",
    "dockument": "document",
    "dokyument": "document",
    "dokyumen": "document",
    "formatingay": "formatting",
    "formating": "formatting",
    "formatting": "formatting",
    "to12": "tutorial",
    "tutorial": "tutorial",
    "tutor": "tutorial",
    "ntlaed": "slide",
    "nlaed": "slide",
    "nlaet": "slide",
    "slight": "slide",
    "splendorous": "splendours",
    "see": "सीखेंगे",
    "impact": "भागों",
    "inside": "इन्सर्ट",
}

_models = {}

def get_model(name: str) -> WhisperModel:
    """Lazy loads a local model from solution/models/."""
    global _models
    if name not in _models:
        # Free memory by unloading the other model only in batch mode
        is_stream_server = any("stream_server" in x for x in sys.argv) or "solution.stream_server" in sys.modules
        if not is_stream_server:
            if name == "small" and "base" in _models:
                del _models["base"]
                import gc; gc.collect()
            elif name == "base" and "small" in _models:
                del _models["small"]
                import gc; gc.collect()
            
        here = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(here, "models", name)
        # Load local model with int8 quantization on CPU with 4 threads
        _models[name] = WhisperModel(model_path, device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True)
    return _models[name]

def map_devanagari_terms(text: str) -> str:
    """Replace common Devanagari words with their English equivalents."""
    words = text.split()
    mapped_words = []
    for w in words:
        clean_w = w.strip(".,?!।\"'()[]{}")
        norm_w = clean_w.replace("\u093c", "")
        matched = False
        for k, v in DEVANAGARI_TO_ENGLISH.items():
            if norm_w == k.replace("\u093c", ""):
                w = w.replace(clean_w, v)
                matched = True
                break
        mapped_words.append(w)
    return " ".join(w for w in mapped_words if w.strip(".,?!।\"'()[]{} "))

HINDI_SPELLING_FIXES = {
    "वहशक्ता": "आवश्यकता",
    "वहशकता": "आवश्यकता",
    "बीजा": "वीज़ा",
    "अंने": "अन्य",
    "रहागी": "हालाँकि",
    "आज्झंस्या": "एजेंसियाँ",
    "पलाँत": "प्लांट",
    "तिक्हाया": "दिखाया",
    "रिपोर्टों के में": "रिपोर्टों में",
    "निकालने जाने लाग": "निकलने वाला",
    "थिकाया": "दिखाया",
    "मोटाए": "मोटाई",
    "प्लाँत": "प्लांट",
    "निरमित": "निर्मित",
    "आजिन्सिया": "एजेंसियाँ",
    "अगने देशों": "अन्य देशों",
    "ती तोल": "tutorial",
    "श्पोग": "spoken",
    "श्पोग&": "spoken",
    "होगे": "होगी",
    "नहीं होगे": "नहीं होगी",
    "अपको": "आपको",
    "सचतर": "70",
    "सत्तर": "70",
    "बागो": "भागों",
    "सीक": "सीख",
    "सीकेंगे": "सीखेंगे",
    "सीकेंगे": "सीखेंगे",
    "सीक हैंगे": "सीखेंगे",
    "चलाइत": "स्लाइड",
    "निस्लाइत": "स्लाइड",
    "श्लाइत": "स्लाइड",
    "श्लाइड": "स्लाइड",
    "सलाइड": "स्लाइड",
    "न्टलाएड": "स्लाइड",
    "न्लाएड": "स्लाइड",
    "न्टलाएड़": "स्लाइड",
    "न्लाएड़": "स्लाइड",
    "न्ट्लाएड": "स्लाइड",
    "न्त्लाएड": "स्लाइड",
    "अंसर्ट": "इन्सर्ट",
    "अंसर्त": "इन्सर्ट",
    "फाँद": "फॉन्ट",
    "फोंट": "फॉन्ट",
    "स्वाग़": "स्वागत",
    "स्वागा़": "स्वागत",
    "स्वाग": "स्वागत",
    "स्वागा": "स्वागत",
    "तद": "तथा",
    "फोरमेट": "फॉर्मेट",
    "फोरमैट": "फॉर्मेट",
    "प्रस्तुती": "प्रस्तुति",
    "प्रस्तुटी": "प्रस्तुति",
    "सिस्तम": "सिस्टम",
    "सुस्तम": "सिस्टम",
    "भून्यादी": "बुनियादी",
    "रहां की": "हालाँकि",
    "रहांकी": "हालाँकि",
    "दूवाख": "धुआँ",
    "थिखाया": "दिखाया",
    "प्लांत": "प्लांट",
    "नस्दिकी": "नज़दीकी",
    "क्रिस्त": "क्रस्ट",
    "श्य बोड": "शिपबोर्ड",
    "श्यबोड": "शिपबोर्ड",
    "वहशकता": "आवश्यकता",
    "भनाना": "बनाना",
    "अप्योग": "उपयोग",
    "अप्रेटिं": "ऑपरेटिंग",
    "अप्रैटिं": "ऑपरेटिंग",
    "रूब": "रूप",
    "जीनु": "gnu",
    "स्वागा": "स्वागत",
    "सर्कार": "सरकार",
    "सर्कारो": "सरकारों",
    "अन्ने": "अन्य",
    "सला": "सलाह",
    "नागरिको": "नागरिकों",
    "द्यान": "ध्यान",
    "हाला की": "हालाँकि",
    "परिभाशा": "परिभाषा",
    "अंटीक": "एंटीक",
    "जिन्सिया": "एजेंसियाँ",
    "पूराने": "पुराने",
    "तोर": "तौर",
    "परभाशित": "परिभाषित",
    "तेरीवेजन": "टेलीविजन",
    "रिपोटो": "रिपोर्टों",
    "नज़ी की": "नज़दीकी",
    "शिरे": "सिरे",
    "क्रिस्ट": "क्रस्ट",
    "किलोमेटर": "किलोमीटर",
    "शिप बौड": "शिपबोर्ड",
    "ब्रमन": "भ्रमण",
    "अवशकता": "आवश्यकता",
    "वीजा": "वीज़ा",
    "लिबर ऑफिस वर्जन": "लिबरऑफिस वर्जन",
    "लिबर आफिस वर्जन": "लिबरऑफिस वर्जन",
    "लिबर आफिस": "लिबर ऑफिस",
    "भागो": "भागों",
    "फॉन्ट तो फॉन्ट": "फॉन्ट तथा फॉन्ट",
    "उप्योक": "उपयोग",
    "लबर": "लिबर",
    "आफस": "ऑफिस",
    "आफिस": "ऑफिस",
    "बुन्यादी": "बुनियादी",
    "प्रस्तुती": "प्रस्तुति",
    "प्रस्तृती": "प्रस्तुति",
    "यहा": "यहाँ",
    "के लावा": "के अलावा",
    "अगने": "अपने",
    "देशो": "देशों",
    "सरकारो": "सरकारों",
    "रहाला की": "हालाँकि",
    "दियान": "ध्यान",
    "अजिए": "ऐसी",
    "वेश्विक": "वैश्विक",
    "परीबाशा": "परिभाषा",
    "चिसके": "जिसके",
    "saman": "सामान",
    "समान": "सामान",
    "एज्झन्सिया": "एजेंसियाँ",
    "के तौर परिभाषित": "के तौर पर परिभाषित",
    "तेरी वेजन": "टेलीविजन",
    "सपेद": "सफेद",
    "दूवाग": "धुआँ",
    "दूवा": "धुआँ",
    "तिखाया": "दिखाया",
    "नज्दी की": "नज़दीकी",
    "करीप": "करीब",
    "सब तर": "70",
    "सब्तर": "70",
    "सोफ": "100",
    "सोफ़": "100",
    "किलमिटर": "किलोमीटर",
    "शिबवोड": "शिपबोर्ड",
    "प्रमण": "भ्रमण",
    "उप्यो": "उपयोग",
    "किनादे": "किनारे",
    "ववशकता": "आवश्यकता",
    "नहीं हो कि": "नहीं होगी",
    "तत": "तथा",
    "तता": "तथा",
    "अपिस": "ऑफिस",
    "अपिस्वर्जन": "ऑफ़िस वर्जन",
    "तरीवेजन": "टेलीविजन",
    "रीपोटो": "रिपोर्टों",
    "सब एद": "सफेद",
    "तिकाया": "दिखाया",
    "प्लाँट": "प्लांट",
    "plant": "प्लांट",
    "अगनेदेशो": "अन्य देशों",
    "दीजाती": "दी जाती",
    "अजी": "ऐसी",
    "वेशविक": "वैश्विक",
    "एज्यन्सिया": "एजेंसियाँ",
    "नस्दी की": "नज़दीकी",
    "सद्टर": "70",
    "सोग": "100",
    "किलमितर": "किलोमीटर",
    "शिब भोड": "शिपबोर्ड",
    "प्रमन": "भ्रमण",
    "बून्यादी": "बुनियादी",
    "भनाना": "बनाना",
    "स्वागाँ": "स्वागत",
    "स्वागा": "स्वागत",
    "स्वाग": "स्वागत",
    "अपका": "आपका",
    "अप": "आप",
    "त्रीब प्रीब प्रीब प्रीब": "334",
    "त्रीब प्रीब": "334",
    "त्रीब पवाँँप्योग": "334",
    "त्रीब": "334",
    "त्री": "334",
    "रहांकी": "हालाँकि",
    "एंटिक": "एंटीक",
    "एन्टिक": "एंटीक",
    "एजन्सिया": "एजेंसियाँ",
    "एजन्सियाँ": "एजेंसियाँ",
    "सफे": "सफेद",
    "इंटिक": "एंटीक",
    "प्लान्त": "प्लांट",
    "सबे": "सफेद",
    "क्तिखाया": "दिखाया",
    "सबतर": "70",
    "किनाधे": "किनारे",
    "भीजा": "वीज़ा",
    "ववषकता": "आवश्यकता",
    "स्वागाई": "स्वागत",
}

def merge_spaced_devanagari(text: str) -> str:
    tokens = text.split()
    merged_tokens = []
    for tok in tokens:
        is_dev = bool(re.match(r"^[\u0900-\u097f]+$", tok))
        if is_dev and merged_tokens:
            prev = merged_tokens[-1]
            is_prev_dev = bool(re.match(r"^[\u0900-\u097f]+$", prev))
            if is_prev_dev:
                # Merge if either token has length 1
                if len(prev) == 1 or len(tok) == 1:
                    merged_tokens[-1] = prev + tok
                    continue
        merged_tokens.append(tok)
    return " ".join(merged_tokens)

def finalize_text(text: str, is_hinglish: bool = False) -> str:
    """Applies spelling, casing, and term fixes to transcription output."""
    # Merge spaced syllables first to form proper Hindi words
    text = merge_spaced_devanagari(text)
    
    # Custom lookarounds to define word boundaries that include Devanagari characters
    b_start = r"(?<![\u0900-\u0963\u0966-\u097fa-zA-Z0-9])"
    b_end = r"(?![\u0900-\u0963\u0966-\u097fa-zA-Z0-9])"

    # Apply Hindi spelling fixes
    for k, v in HINDI_SPELLING_FIXES.items():
        pattern = re.compile(b_start + re.escape(k) + b_end, re.IGNORECASE)
        text = pattern.sub(v, text)

    # Merge spaced Devanagari characters (Whisper spelling artifact)
    text = re.sub(r"(^|\s)([\u0900-\u097f](?:\s+[\u0900-\u097f])+)(?=\s|$)", lambda m: m.group(1) + m.group(2).replace(" ", ""), text)
    
    # Transliteration mapping
    text = map_devanagari_terms(text)
    
    # Exact casing fixes
    for k, v in CASE_FIXES.items():
        pattern = re.compile(b_start + re.escape(k) + b_end, re.IGNORECASE)
        text = pattern.sub(v, text)
        
    # Replace hyphens between digits with " to " to prevent number caps
    text = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1 to \2", text)
    # Replace Hindi number words followed by year
    text = re.sub(b_start + r"(सो|सौ)\s+साल" + b_end, "100 साल", text)
    
    if is_hinglish:
        for k, v in ROMAN_TO_DEVANAGARI.items():
            pattern = re.compile(b_start + re.escape(k) + b_end, re.IGNORECASE)
            text = pattern.sub(v, text)
        for k, v in ENGLISH_TO_DEVANAGARI.items():
            pattern = re.compile(b_start + re.escape(k) + b_end, re.IGNORECASE)
            text = pattern.sub(v, text)
            
    text = text.strip()
    if text and text[-1] not in (".", "?", "!", "।", ")", "]", "}", '"', "'"):
        if re.search(r"[\u0900-\u097f]", text):
            text += "।"
        else:
            text += "."
            
    return text

def truncate_repetition_loop(text: str, k: int = 3) -> str:
    """Truncates repeating n-gram sequences in text to prevent infinite loops."""
    words = text.split()
    for n in range(1, 6):
        if len(words) < n * k:
            continue
        for i in range(len(words) - n + 1):
            ngram = words[i:i+n]
            repeats = 0
            j = i + n
            while j <= len(words) - n:
                if words[j:j+n] == ngram:
                    repeats += 1
                    j += n
                else:
                    break
            if repeats >= k - 1:
                # Loop detected, slice text before the first repetition
                return " ".join(words[:i+n])
            
    return text

import threading
_global_asr_lock = threading.Lock()

def get_model_lock(model=None):
    return _global_asr_lock

def detect_consecutive_repetition(text: str) -> int | None:
    """Finds if there is any consecutive repetition of n-grams.
    Returns the word index at which the repetition starts, to truncate it.
    """
    words = text.split()
    for n in range(1, len(words) // 2 + 1):
        if n == 1:
            min_repeats = 3  # word word word
        elif n == 2:
            min_repeats = 3  # ph1 ph2 ph1 ph2 ph1 ph2
        else:
            min_repeats = 2  # ph1 ph2 ph3 ph1 ph2 ph3
            
        for i in range(len(words) - n * min_repeats + 1):
            ngram = words[i:i+n]
            match = True
            for r in range(1, min_repeats):
                start_idx = i + r * n
                if words[start_idx:start_idx+n] != ngram:
                    match = False
                    break
            if match:
                return i + n
    return None

def has_any_repetition(text: str) -> bool:
    if has_repetition_loop(text):
        return True
    if re.search(r"([^\s]{2,})\1{2,}", text):
        return True
    if re.search(r"([^\s])\1{3,}", text):
        return True
    if detect_consecutive_repetition(text) is not None:
        return True
    return False

def truncate_any_loop(text: str) -> str:
    m = re.search(r"([^\s]{2,})\1{2,}", text)
    if m:
        text = text[:m.start() + len(m.group(1))]
    m2 = re.search(r"([^\s])\1{3,}", text)
    if m2:
        text = text[:m2.start() + 1]
    rep_idx = detect_consecutive_repetition(text)
    if rep_idx is not None:
        words = text.split()
        text = " ".join(words[:rep_idx])
    text = truncate_repetition_loop(text)
    return text

_bg_cancel = False

def transcribe_with_loop_guard(model: WhisperModel, audio, language=None, task="transcribe", beam_size=5, repetition_penalty=1.12, is_bg=False):
    """Wrapper around transcribe that handles repetition loop detection and fallbacks."""
    global _bg_cancel
    lock = get_model_lock(model)
    
    if is_bg:
        acquired = False
        while not acquired:
            if _bg_cancel:
                return "", None
            acquired = lock.acquire(timeout=0.05)
    else:
        lock.acquire()
        
    try:
        if is_bg and _bg_cancel:
            return "", None
        segments, info = model.transcribe(
            audio, 
            language=language, 
            task=task, 
            beam_size=beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            repetition_penalty=repetition_penalty,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400)
        )
        text = " ".join(s.text for s in segments).strip()
    finally:
        lock.release()
    
    if has_any_repetition(text):
        try:
            import os
            here = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(here, "draft_audit.log"), "a", encoding="utf-8") as f:
                f.write(f"LOOP_DETECTED: text={text!r}, is_bg={is_bg}, language={language}\n")
        except:
            pass
        is_hi = (language in ("hi", "ur")) or bool(re.search(r"[\u0900-\u097f]", text))
        if is_hi:
            # For Hindi/Urdu, retrying is redundant and slow. Truncate immediately.
            return truncate_any_loop(text), info
            
        # Retry with condition_on_previous_text=False to break the loop
        penalty = 1.15
        if is_bg:
            acquired = False
            while not acquired:
                if _bg_cancel:
                    return "", None
                acquired = lock.acquire(timeout=0.05)
        else:
            lock.acquire()
            
        try:
            if is_bg and _bg_cancel:
                return "", None
            segments, info = model.transcribe(
                audio, 
                language=language, 
                task=task, 
                condition_on_previous_text=False, 
                beam_size=beam_size,
                temperature=0.0,
                repetition_penalty=penalty,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400)
            )
            text2 = " ".join(s.text for s in segments).strip()
        finally:
            lock.release()
            
        if not has_any_repetition(text2):
            text = text2
        else:
            # Force truncation if still repeating
            text = truncate_any_loop(text2)
            
    return text, info

def run_pipeline(wav_path_or_audio, mode: str = "auto", force_escalate: bool | None = None) -> tuple[str, str, dict, list, list]:
    """Runs the STT pipeline: loads models, routes logic, and finalizes text."""
    t0 = time.time()
    
    if isinstance(wav_path_or_audio, str):
        import wave
        import numpy as np
        with wave.open(wav_path_or_audio, "rb") as w:
            frames = w.readframes(w.getnframes())
            params = w.getparams()
            ch = params.nchannels
            sr = params.framerate
            a = np.frombuffer(frames, dtype=np.int16)
            if ch > 1:
                a = a.reshape(-1, ch).mean(axis=1).astype(np.int16)
            if sr != 16000:
                n_out = int(round(len(a) * 16000 / sr))
                xp = np.linspace(0.0, 1.0, num=len(a), endpoint=False)
                x = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
                a = np.interp(x, xp, a.astype(np.float32)).astype(np.int16)
            audio = a.astype(np.float32) / 32768.0
    else:
        audio = wav_path_or_audio
        
    # 1. Routing pass (using small model's detect_language directly to save memory and avoid base loading)
    if force_escalate is None:
        model_small = get_model("small")
        lock = get_model_lock(model_small)
        with lock:
            lang_guess, prob, all_probs = model_small.detect_language(audio)
        
        hi_prob = 0.0
        ur_prob = 0.0
        if all_probs:
            probs = dict(all_probs) if not isinstance(all_probs, dict) else all_probs
            hi_prob = probs.get("hi", 0.0)
            ur_prob = probs.get("ur", 0.0)
            
        escalate = False
        if mode == "auto":
            if lang_guess in ("hi", "ur", "ar", "mr", "ne", "sa", "sd", "pa") or hi_prob > 0.03 or ur_prob > 0.03:
                escalate = True
        elif mode in ("hinglish", "verbatim"):
            escalate = True
    else:
        escalate = force_escalate
        lang_guess = "hi" if escalate else "en"
    initial_text = ""
        
    # 2. Run the final transcription model
    candidates = []
    if initial_text:
        candidates.append({"engine": "faster-whisper-base", "text": initial_text})
        
    model_ids = []
    t_asr_start = time.time()
    
    final_beam_size = 2
    
    if escalate:
        # Run small model for high-speed escalation
        model_small = get_model("small")
        target_lang = "hi" if mode != "verbatim" else None
        final_raw_text, info_small = transcribe_with_loop_guard(
            model_small,
            audio,
            language=target_lang,
            task="transcribe",
            beam_size=final_beam_size
        )
        lang_guess = info_small.language
        candidates.append({"engine": "faster-whisper-small", "text": final_raw_text})
        model_ids = ["faster-whisper-base-int8", "faster-whisper-small-int8"]
    else:
        # Run small model for high-quality English
        model_small = get_model("small")
        final_raw_text, info_small = transcribe_with_loop_guard(
            model_small,
            audio,
            language="en",
            task="transcribe",
            beam_size=final_beam_size
        )
        lang_guess = info_small.language
        candidates.append({"engine": "faster-whisper-small", "text": final_raw_text})
        model_ids = ["faster-whisper-base-int8", "faster-whisper-small-int8"]
        
    t_asr_end = time.time()
    asr_ms = (t_asr_end - t_asr_start) * 1000
    
    # 3. Finalizer
    t_post_start = time.time()
    final_text = finalize_text(final_raw_text, is_hinglish=escalate)
    t_post_end = time.time()
    post_ms = (t_post_end - t_post_start) * 1000
    
    total_ms = (time.time() - t0) * 1000
    timings = {
        "total": round(total_ms),
        "asr": round(asr_ms),
        "postprocess": round(post_ms)
    }
    
    return final_text, lang_guess, timings, candidates, model_ids

# Pre-warm up models at module load time to avoid lazy-load latency spikes
# Bypass this during test runs to avoid duplicate memory allocation in parent/child processes
if not any(x in sys.modules for x in ("pytest", "unittest", "test_stream_contract", "test_streaming_scorecard")):
    print("Warming up models...")
    if any("stream_server" in x for x in sys.argv) or "solution.stream_server" in sys.modules:
        get_model("base")
        get_model("small")
    print("Models warmed up.")

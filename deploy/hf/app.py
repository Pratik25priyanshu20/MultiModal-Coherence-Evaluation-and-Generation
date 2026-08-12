"""
Multimodal Coherence AI — Hugging Face Spaces Demo

Live demonstration of multimodal generation + coherence evaluation.
Enter a scene description and the system produces coherent text, image,
and audio with real-time MSCI scoring.

Pipeline: Groq LLM (text) + Pollinations (image) + ElevenLabs (audio SFX) with CLIP/CLAP retrieval fallback
Planning modes: direct, planner, council (3-way), extended_prompt (3x tokens)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

IMAGE_SIM_THRESHOLD = 0.20
AUDIO_SIM_THRESHOLD = 0.10

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

.block-container { padding-top: 1.2rem !important; max-width: 1200px; }
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

.hero-wrap { text-align: center; padding: 1.5rem 0 1rem; }
.hero-title {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -0.03em;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.35rem;
}
.hero-sub {
    font-size: 1rem; color: #94a3b8; max-width: 600px;
    margin: 0 auto; line-height: 1.6;
}
.hero-sub b { color: #c4b5fd; }

.stTextArea textarea {
    border-radius: 14px !important;
    border: 1.5px solid rgba(129,140,248,0.25) !important;
    font-size: 0.95rem !important; padding: 0.9rem 1rem !important;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus {
    border-color: rgba(129,140,248,0.6) !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.1) !important;
}

.chip-row { display: flex; gap: 0.4rem; flex-wrap: wrap; align-items: center; padding-top: 0.3rem; }
.chip {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.22rem 0.7rem; border-radius: 20px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.03em;
}
.chip-purple { background: rgba(129,140,248,0.14); color: #a5b4fc; }
.chip-pink   { background: rgba(244,114,182,0.14); color: #f9a8d4; }
.chip-green  { background: rgba(52,211,153,0.14);  color: #6ee7b7; }
.chip-amber  { background: rgba(251,191,36,0.12);  color: #fcd34d; }
.chip-dot { width: 6px; height: 6px; border-radius: 50%; }
.chip-dot-purple { background: #818cf8; }
.chip-dot-pink   { background: #f472b6; }
.chip-dot-green  { background: #34d399; }
.chip-dot-amber  { background: #fbbf24; }

.scores-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem; margin: 0.5rem 0 0.3rem;
}
@media (max-width: 768px) { .scores-grid { grid-template-columns: repeat(2, 1fr); } }
.sc {
    border-radius: 16px; padding: 1.1rem 0.8rem; text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.02);
    backdrop-filter: blur(10px);
    position: relative; overflow: hidden;
}
.sc::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 16px 16px 0 0;
}
.sc-high::before { background: linear-gradient(90deg, #10b981, #34d399); }
.sc-mid::before  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.sc-low::before  { background: linear-gradient(90deg, #ef4444, #fb7185); }
.sc-class::before { background: linear-gradient(90deg, #818cf8, #c084fc); }
.sc-lbl {
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #64748b; margin-bottom: 0.4rem; font-weight: 600;
}
.sc-val {
    font-size: 1.9rem; font-weight: 700; line-height: 1.1;
    font-family: 'JetBrains Mono', monospace;
}
.sc-high .sc-val { color: #34d399; }
.sc-mid  .sc-val { color: #fbbf24; }
.sc-low  .sc-val { color: #fb7185; }
.sc-class .sc-val { font-size: 1.15rem; font-family: 'Inter', sans-serif; color: #c4b5fd; }
.sc-badge {
    display: inline-block; margin-top: 0.35rem; padding: 0.15rem 0.55rem;
    border-radius: 20px; font-size: 0.6rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.07em;
}
.sc-high .sc-badge { background: rgba(52,211,153,0.12); color: #34d399; }
.sc-mid  .sc-badge { background: rgba(251,191,36,0.12); color: #fbbf24; }
.sc-low  .sc-badge { background: rgba(251,113,133,0.12); color: #fb7185; }
.sc-class .sc-badge { background: rgba(196,181,253,0.12); color: #c4b5fd; }

.sec-label {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em;
    font-weight: 700; margin-bottom: 0.6rem; padding-bottom: 0.35rem;
    border-bottom: 2px solid rgba(129,140,248,0.15); color: #818cf8;
}
.text-card {
    border-radius: 14px; padding: 1.1rem 1.2rem;
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    font-size: 0.9rem; line-height: 1.75; color: #cbd5e1;
}
.timing {
    display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center;
    padding: 0.4rem 0.8rem; border-radius: 10px;
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04);
    font-size: 0.72rem; color: #64748b; margin: 0.4rem 0;
}
.timing span { white-space: nowrap; }
.timing .t-total { color: #a5b4fc; font-weight: 700; }
.timing .t-sep { color: rgba(255,255,255,0.08); }

.warn-banner {
    border-radius: 12px; padding: 0.7rem 1rem; margin-bottom: 0.6rem;
    border-left: 3px solid #fbbf24; font-size: 0.82rem; color: #fcd34d;
    background: rgba(251,191,36,0.05);
}
.warn-banner b { color: #fde68a; }

.sb { margin: 0.35rem 0; }
.sb-top { display: flex; justify-content: space-between; font-size: 0.68rem; color: #64748b; margin-bottom: 0.15rem; }
.sb-top .sb-v { font-family: 'JetBrains Mono', monospace; font-weight: 600; }
.sb-track { height: 5px; border-radius: 3px; background: rgba(255,255,255,0.05); overflow: hidden; }
.sb-fill { height: 100%; border-radius: 3px; }
.sbf-g { background: linear-gradient(90deg, #10b981, #34d399); }
.sbf-y { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.sbf-r { background: linear-gradient(90deg, #ef4444, #fb7185); }

.welcome { text-align: center; padding: 4rem 2rem; color: #475569; }
.welcome-icons { font-size: 3.5rem; margin-bottom: 0.8rem; letter-spacing: 0.3rem; }
.welcome-text { font-size: 1.05rem; color: #64748b; }
.welcome-hint { font-size: 0.82rem; color: #475569; margin-top: 0.3rem; }

section[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
.sidebar-info {
    font-size: 0.72rem; color: #64748b; line-height: 1.6;
    padding: 0.8rem; border-radius: 10px;
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04);
}
.sidebar-info b { color: #94a3b8; }
</style>
"""

# ---------------------------------------------------------------------------
# Example prompts
# ---------------------------------------------------------------------------
EXAMPLE_PROMPTS = {
    "en": {
        "Nature": [
            "A peaceful forest at dawn with birdsong and morning mist",
            "A field of golden wheat under a warm summer sunset",
            "A dense jungle with exotic birds calling from the canopy",
        ],
        "Urban": [
            "A bustling city street at night with neon lights and traffic",
            "A quiet alley in an old town with distant footsteps echoing",
            "A cafe terrace on a busy boulevard with clinking glasses",
        ],
        "Water": [
            "Ocean waves crashing on a sandy beach at sunset",
            "Rain falling on a pond with ripples spreading across the surface",
            "A mountain stream flowing over rocks through a pine forest",
        ],
        "Mixed": [
            "A lighthouse on a cliff during a thunderstorm at night",
            "A bonfire on a beach with waves and guitar music at night",
            "A train passing through countryside with distant church bells",
        ],
    },
    "de": {
        "Natur": [
            "Ein friedlicher Wald bei Sonnenaufgang mit Vogelgesang und Morgennebel",
            "Ein goldenes Weizenfeld unter einem warmen Sommerabend",
            "Ein dichter Dschungel mit exotischen V\u00f6geln im Bl\u00e4tterdach",
        ],
        "Stadt": [
            "Eine belebte Stra\u00dfe bei Nacht mit Neonlichtern und Verkehr",
            "Eine ruhige Gasse in einer Altstadt mit fernen Schritten",
            "Eine Caf\u00e9-Terrasse an einem belebten Boulevard mit klinkenden Gl\u00e4sern",
        ],
        "Wasser": [
            "Meereswellen am Sandstrand bei Sonnenuntergang",
            "Regen f\u00e4llt auf einen Teich mit sich ausbreitenden Wellen",
            "Ein Bergbach flie\u00dft \u00fcber Felsen durch einen Kiefernwald",
        ],
        "Gemischt": [
            "Ein Leuchtturm auf einer Klippe w\u00e4hrend eines Gewitters bei Nacht",
            "Ein Lagerfeuer am Strand mit Wellen und Gitarrenmusik bei Nacht",
            "Ein Zug f\u00e4hrt durch die Landschaft mit fernen Kirchenglocken",
        ],
    },
}
DOMAIN_ICONS = {"nature": "\U0001f33f", "urban": "\U0001f3d9\ufe0f", "water": "\U0001f30a", "mixed": "\U0001f310", "other": "\U0001f4cd"}

# ---------------------------------------------------------------------------
# Kid Mode — example prompts (German, fun themes for children)
# ---------------------------------------------------------------------------
KID_EXAMPLE_PROMPTS = {
    "de": {
        "\U0001f47e Abenteuer": [
            "Pikachu in einem magischen Wald bei Sonnenuntergang",
            "Ein Minecraft-Dorf auf einer Insel mitten im Ozean",
            "Ein kleiner Drache fliegt \u00fcber eine Burg bei Nacht",
            "Ein Weltraumabenteuer mit Raketen und bunten Planeten",
        ],
        "\U0001f43e Tiere": [
            "Ein freundlicher Hund rettet ein K\u00e4tzchen im Regen",
            "Dinosaurier spielen Fu\u00dfball auf einer sonnigen Wiese",
            "Ein Einhorn galoppiert \u00fcber einen leuchtenden Regenbogen",
            "Pinguine machen eine Schneeballschlacht am S\u00fcdpol",
            "Ein kleiner Fuchs entdeckt einen geheimen Garten",
        ],
        "\u2728 Fantasie": [
            "Ein Zauberer braut einen glitzernden Trank in einem Schloss",
            "Eine Fee fliegt durch einen Wald voller leuchtender Pilze",
            "Ein verzaubertes Baumhaus in den Wolken mit Regenbogenbr\u00fccke",
            "Ein Roboter und ein Teddy gehen zusammen auf Schatzsuche",
            "Ein magischer Unterwasserpalast mit sprechenden Fischen",
        ],
        "\U0001f602 Lustig": [
            "Eine Katze f\u00e4hrt Skateboard durch eine bunte Stadt",
            "Aliens landen im Schulgarten und spielen Verstecken",
            "Ein Elefant versucht sich auf einem Trampolin",
            "Ein Schneemann isst Eis am Strand im Sommer",
            "Monster unter dem Bett machen eine Pyjamaparty",
        ],
        "\U0001f3ae Spielwelt": [
            "Super Mario springt durch eine Welt aus S\u00fc\u00dfigkeiten",
            "Ein Ritter k\u00e4mpft gegen einen freundlichen Drachen",
            "Eine Unterwasser-Rennstrecke mit U-Booten und Delfinen",
            "Ein Baumhaus-Dorf im Dschungel mit H\u00e4ngebr\u00fccken",
            "Tiere bauen zusammen eine riesige Sandburg am Meer",
        ],
    },
    "en": {
        "\U0001f47e Adventure": [
            "Pikachu in a magical forest at sunset",
            "A Minecraft village on an island in the middle of the ocean",
            "A little dragon flying over a castle at night",
            "A space adventure with rockets and colorful planets",
        ],
        "\U0001f43e Animals": [
            "A friendly dog rescuing a kitten in the rain",
            "Dinosaurs playing football on a sunny meadow",
            "A unicorn galloping over a glowing rainbow",
            "Penguins having a snowball fight at the South Pole",
            "A little fox discovering a secret garden",
        ],
        "\u2728 Fantasy": [
            "A wizard brewing a sparkling potion in a castle",
            "A fairy flying through a forest of glowing mushrooms",
            "An enchanted treehouse in the clouds with a rainbow bridge",
            "A robot and a teddy bear going on a treasure hunt together",
            "A magical underwater palace with talking fish",
        ],
        "\U0001f602 Funny": [
            "A cat riding a skateboard through a colorful city",
            "Aliens landing in the school garden and playing hide and seek",
            "An elephant trying to jump on a trampoline",
            "A snowman eating ice cream at the beach in summer",
            "Monsters under the bed having a pajama party",
        ],
        "\U0001f3ae Game World": [
            "Super Mario jumping through a world made of candy",
            "A knight fighting a friendly dragon",
            "An underwater race track with submarines and dolphins",
            "A treehouse village in the jungle with rope bridges",
            "Animals building a giant sandcastle at the beach",
        ],
    },
}

# ---------------------------------------------------------------------------
# Kid Mode — CSS theme (bright, bubbly, playful)
# ---------------------------------------------------------------------------
KID_CSS = """
<style>
/* ============================================================
   KID MODE — Full theme override
   ============================================================ */

/* Kill the top gap */
.block-container { padding-top: 0.5rem !important; }
header[data-testid="stHeader"] { display: none !important; }

/* Force light colorful background on EVERYTHING */
.stApp, .stApp > div, .main, .main .block-container,
[data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"],
section.main, section.main > div {
    background: linear-gradient(170deg, #dbeafe 0%, #fce7f3 35%, #fef3c7 65%, #dcfce7 100%) !important;
    color: #1e293b !important;
}
/* Sidebar light theme */
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #ede9fe 0%, #fce7f3 100%) !important;
    color: #1e293b !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] p {
    color: #334155 !important;
}
/* Force dark text everywhere */
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div,
.stTextArea textarea, label, .stSelectbox label {
    color: #1e293b !important;
}
.stTextArea textarea {
    background: rgba(255,255,255,0.85) !important;
    border: 2px solid #c4b5fd !important;
    border-radius: 18px !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 4px rgba(139,92,246,0.15) !important;
}
/* Status containers */
[data-testid="stStatusWidget"] {
    background: rgba(255,255,255,0.6) !important;
    border-radius: 14px !important;
}

/* Floating background elements */
.kid-bg {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 0; overflow: hidden;
}
.kid-bg-item {
    position: absolute; opacity: 0.15;
    animation: kid-float linear infinite;
}
@keyframes kid-float {
    0%   { transform: translateY(105vh) rotate(0deg) scale(0.8); opacity: 0; }
    8%   { opacity: 0.35; }
    92%  { opacity: 0.35; }
    100% { transform: translateY(-10vh) rotate(360deg) scale(1.1); opacity: 0; }
}
/* Twinkle for stars */
@keyframes kid-twinkle {
    0%, 100% { opacity: 0.15; transform: scale(0.8); }
    50% { opacity: 0.5; transform: scale(1.2); }
}
.kid-star-fixed {
    position: absolute; pointer-events: none;
    animation: kid-twinkle ease-in-out infinite;
}
/* Clouds */
.kid-cloud {
    position: absolute; pointer-events: none; opacity: 0.18;
    width: 120px; height: 50px; background: white;
    border-radius: 50px; animation: kid-drift linear infinite;
}
.kid-cloud::before {
    content: ''; position: absolute; background: white; border-radius: 50%;
    width: 55px; height: 55px; top: -25px; left: 20px;
}
.kid-cloud::after {
    content: ''; position: absolute; background: white; border-radius: 50%;
    width: 40px; height: 40px; top: -18px; left: 55px;
}
@keyframes kid-drift {
    0%   { transform: translateX(-150px); }
    100% { transform: translateX(calc(100vw + 150px)); }
}

/* Hero — big colorful title */
.kid-hero {
    text-align: center; padding: 0.8rem 0 0.3rem; position: relative; z-index: 1;
}
.kid-hero-title {
    font-size: 3.2rem; font-weight: 900; letter-spacing: -0.02em;
    background: linear-gradient(135deg, #ec4899, #f97316, #eab308, #22c55e, #3b82f6, #8b5cf6);
    background-size: 300% 300%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: kid-gradient 4s ease infinite;
    text-shadow: none;
}
@keyframes kid-gradient {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.kid-hero-sub {
    font-size: 1.15rem; color: #475569; margin-top: 0.2rem; font-weight: 500;
}
.kid-hero-sub b { color: #7c3aed; }

/* Mascots — bigger, animated, with speech bubbles */
.kid-mascot-row {
    display: flex; justify-content: center; gap: 2rem; margin: 0.8rem 0 0.5rem;
    position: relative; z-index: 1;
}
.kid-mascot {
    display: flex; flex-direction: column; align-items: center;
    padding: 0.8rem 1.2rem 0.5rem; border-radius: 24px;
    background: rgba(255,255,255,0.9);
    border: 3px solid rgba(255,255,255,1);
    box-shadow: 0 8px 30px rgba(0,0,0,0.08), 0 2px 8px rgba(139,92,246,0.1);
    transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    cursor: default; position: relative;
    min-width: 105px;
}
.kid-mascot:hover {
    transform: scale(1.12) rotate(-3deg);
    box-shadow: 0 12px 40px rgba(139,92,246,0.25);
}
.kid-mascot svg { display: block; margin: 0 auto; }
.kid-mascot-name {
    font-size: 0.9rem; font-weight: 800; margin-top: 0.15rem;
    letter-spacing: 0.04em;
}
.kid-mascot:nth-child(1) .kid-mascot-name { color: #3b82f6; }
.kid-mascot:nth-child(2) .kid-mascot-name { color: #ec4899; }
.kid-mascot:nth-child(3) .kid-mascot-name { color: #f97316; }
/* Continuous gentle bounce */
.kid-mascot:nth-child(1) { animation: kid-bob 2s ease-in-out infinite; }
.kid-mascot:nth-child(2) { animation: kid-bob 2s ease-in-out 0.3s infinite; }
.kid-mascot:nth-child(3) { animation: kid-bob 2s ease-in-out 0.6s infinite; }
@keyframes kid-bob {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}
.kid-mascot:hover { animation: none; }
/* Speech bubble */
.kid-speech {
    position: absolute; top: -32px; left: 50%; transform: translateX(-50%);
    background: #fef3c7; color: #92400e; font-size: 0.65rem; font-weight: 700;
    padding: 3px 10px; border-radius: 12px; white-space: nowrap;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    opacity: 0; transition: opacity 0.2s;
}
.kid-speech::after {
    content: ''; position: absolute; bottom: -5px; left: 50%; margin-left: -5px;
    border-left: 5px solid transparent; border-right: 5px solid transparent;
    border-top: 5px solid #fef3c7;
}
.kid-mascot:hover .kid-speech { opacity: 1; }

/* Score cards — kid version */
.kid-scores {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem; margin: 0.6rem 0; position: relative; z-index: 1;
}
@media (max-width: 768px) { .kid-scores { grid-template-columns: repeat(2, 1fr); } }
.kid-sc {
    border-radius: 22px; padding: 1.1rem 0.8rem; text-align: center;
    background: rgba(255,255,255,0.85);
    border: 2.5px solid rgba(255,255,255,1);
    box-shadow: 0 6px 24px rgba(0,0,0,0.06);
    position: relative; overflow: hidden;
    animation: kid-pop 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}
.kid-sc:nth-child(1) { animation-delay: 0s; }
.kid-sc:nth-child(2) { animation-delay: 0.1s; }
.kid-sc:nth-child(3) { animation-delay: 0.2s; }
.kid-sc:nth-child(4) { animation-delay: 0.3s; }
@keyframes kid-pop {
    0% { transform: scale(0.7); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}
.kid-sc::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 5px;
    border-radius: 22px 22px 0 0;
}
.kid-sc-great::before { background: linear-gradient(90deg, #22c55e, #06b6d4); }
.kid-sc-ok::before { background: linear-gradient(90deg, #f59e0b, #f97316); }
.kid-sc-low::before { background: linear-gradient(90deg, #ef4444, #ec4899); }
.kid-sc-main::before { background: linear-gradient(90deg, #8b5cf6, #ec4899, #f97316, #eab308); background-size: 200%; animation: kid-gradient 3s ease infinite; }
.kid-sc-lbl {
    font-size: 0.72rem; font-weight: 800; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.kid-sc-stars { font-size: 1.8rem; margin: 0.3rem 0; line-height: 1.1; }
.kid-sc-emoji { font-size: 2.4rem; margin: 0.15rem 0; }
.kid-sc-val {
    font-size: 0.7rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;
}

/* Verdict banner */
.kid-verdict {
    text-align: center; font-size: 1.4rem; font-weight: 800;
    color: #334155; margin: 0.4rem 0 0.6rem;
    animation: kid-pop 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}

/* Section labels */
.kid-sec-label {
    font-size: 0.85rem; font-weight: 900; letter-spacing: 0.06em;
    text-transform: uppercase; color: #7c3aed !important;
    padding-bottom: 0.35rem; border-bottom: 3px solid #c4b5fd;
    margin-bottom: 0.6rem;
}
.kid-text-card {
    border-radius: 20px; padding: 1.2rem 1.3rem;
    background: rgba(255,255,255,0.8);
    border: 2px solid rgba(255,255,255,1);
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    font-size: 0.95rem; line-height: 1.8; color: #334155 !important;
}

.kid-timing {
    display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center;
    padding: 0.45rem 0.9rem; border-radius: 16px;
    background: rgba(255,255,255,0.6);
    border: 2px solid rgba(255,255,255,0.9);
    font-size: 0.72rem; color: #64748b !important; margin: 0.4rem 0;
}
.kid-timing span { color: #64748b !important; }
.kid-timing .t-total { color: #7c3aed !important; font-weight: 700; }
.kid-timing .t-sep { color: #cbd5e1 !important; }

/* Warn banner */
.kid-warn {
    border-radius: 16px; padding: 0.8rem 1.1rem; margin-bottom: 0.6rem;
    border-left: 4px solid #f97316; font-size: 0.85rem; color: #9a3412 !important;
    background: rgba(255,237,213,0.7);
}

/* Button override — primary (Let's Go / Generate) */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #8b5cf6, #ec4899) !important;
    border: none !important; border-radius: 16px !important;
    font-weight: 800 !important; font-size: 1.05rem !important;
    padding: 0.6rem 1.5rem !important;
    box-shadow: 0 4px 15px rgba(139,92,246,0.3) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: scale(1.03) !important;
    box-shadow: 0 6px 25px rgba(139,92,246,0.4) !important;
}

/* Button override — secondary (prompt suggestion buttons in sidebar) */
.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.85) !important;
    color: #4c1d95 !important;
    border: 2px solid #c4b5fd !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 0.8rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind="primary"]):hover {
    background: linear-gradient(135deg, #ede9fe, #fce7f3) !important;
    border-color: #8b5cf6 !important;
    color: #3b0764 !important;
    transform: scale(1.02) !important;
    box-shadow: 0 3px 12px rgba(139,92,246,0.2) !important;
}

/* Expander headers in sidebar — light and readable */
section[data-testid="stSidebar"] details summary {
    background: rgba(255,255,255,0.6) !important;
    color: #4c1d95 !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}

/* Divider */
hr { border-color: rgba(139,92,246,0.15) !important; }
</style>
"""

# ---------------------------------------------------------------------------
# Kid Mode — mascot HTML, star ratings, emoji feedback
# ---------------------------------------------------------------------------

MASCOT_HTML = """
<!-- Rich floating background -->
<div class="kid-bg">
    <!-- Wave 1: floating emoji rising (spread across page) -->
    <div class="kid-bg-item" style="font-size:30px;left:2%;animation-duration:14s;">\u2b50</div>
    <div class="kid-bg-item" style="font-size:24px;left:8%;animation-duration:18s;animation-delay:2s;">\U0001f98b</div>
    <div class="kid-bg-item" style="font-size:26px;left:14%;animation-duration:16s;animation-delay:5s;">\U0001f49c</div>
    <div class="kid-bg-item" style="font-size:20px;left:20%;animation-duration:22s;animation-delay:1s;">\U0001f680</div>
    <div class="kid-bg-item" style="font-size:32px;left:26%;animation-duration:13s;animation-delay:3s;">\u2728</div>
    <div class="kid-bg-item" style="font-size:22px;left:32%;animation-duration:19s;animation-delay:7s;">\U0001f338</div>
    <div class="kid-bg-item" style="font-size:28px;left:38%;animation-duration:15s;animation-delay:4s;">\U0001f31f</div>
    <div class="kid-bg-item" style="font-size:18px;left:44%;animation-duration:20s;animation-delay:0s;">\U0001f984</div>
    <div class="kid-bg-item" style="font-size:26px;left:50%;animation-duration:17s;animation-delay:6s;">\U0001f308</div>
    <div class="kid-bg-item" style="font-size:24px;left:56%;animation-duration:14s;animation-delay:2s;">\U0001f49b</div>
    <div class="kid-bg-item" style="font-size:20px;left:62%;animation-duration:21s;animation-delay:8s;">\U0001f33c</div>
    <div class="kid-bg-item" style="font-size:30px;left:68%;animation-duration:16s;animation-delay:1s;">\u2b50</div>
    <div class="kid-bg-item" style="font-size:22px;left:74%;animation-duration:18s;animation-delay:5s;">\U0001f98b</div>
    <div class="kid-bg-item" style="font-size:28px;left:80%;animation-duration:13s;animation-delay:3s;">\u2728</div>
    <div class="kid-bg-item" style="font-size:24px;left:86%;animation-duration:20s;animation-delay:9s;">\U0001f49a</div>
    <div class="kid-bg-item" style="font-size:18px;left:92%;animation-duration:15s;animation-delay:4s;">\U0001f30d</div>
    <div class="kid-bg-item" style="font-size:26px;left:97%;animation-duration:17s;animation-delay:0s;">\U0001f680</div>
    <!-- Wave 2: offset for constant density -->
    <div class="kid-bg-item" style="font-size:22px;left:5%;animation-duration:19s;animation-delay:10s;">\U0001f33c</div>
    <div class="kid-bg-item" style="font-size:28px;left:15%;animation-duration:15s;animation-delay:11s;">\U0001f49b</div>
    <div class="kid-bg-item" style="font-size:18px;left:25%;animation-duration:21s;animation-delay:9s;">\U0001f984</div>
    <div class="kid-bg-item" style="font-size:26px;left:35%;animation-duration:16s;animation-delay:12s;">\u2b50</div>
    <div class="kid-bg-item" style="font-size:24px;left:45%;animation-duration:18s;animation-delay:8s;">\U0001f98b</div>
    <div class="kid-bg-item" style="font-size:20px;left:55%;animation-duration:14s;animation-delay:13s;">\U0001f308</div>
    <div class="kid-bg-item" style="font-size:30px;left:65%;animation-duration:20s;animation-delay:10s;">\u2728</div>
    <div class="kid-bg-item" style="font-size:22px;left:75%;animation-duration:17s;animation-delay:11s;">\U0001f338</div>
    <div class="kid-bg-item" style="font-size:26px;left:85%;animation-duration:13s;animation-delay:14s;">\U0001f49a</div>
    <div class="kid-bg-item" style="font-size:24px;left:95%;animation-duration:19s;animation-delay:9s;">\U0001f31f</div>
    <!-- Wave 3: more for richness -->
    <div class="kid-bg-item" style="font-size:20px;left:10%;animation-duration:17s;animation-delay:15s;">\U0001f680</div>
    <div class="kid-bg-item" style="font-size:26px;left:30%;animation-duration:14s;animation-delay:16s;">\U0001f338</div>
    <div class="kid-bg-item" style="font-size:22px;left:50%;animation-duration:19s;animation-delay:14s;">\U0001f984</div>
    <div class="kid-bg-item" style="font-size:28px;left:70%;animation-duration:15s;animation-delay:17s;">\U0001f49c</div>
    <div class="kid-bg-item" style="font-size:24px;left:90%;animation-duration:18s;animation-delay:15s;">\U0001f33c</div>
    <!-- Twinkling stars (fixed) -->
    <div class="kid-star-fixed" style="font-size:18px;top:5%;left:8%;animation-duration:2.5s;">\u2b50</div>
    <div class="kid-star-fixed" style="font-size:14px;top:12%;left:30%;animation-duration:3s;animation-delay:0.5s;">\u2b50</div>
    <div class="kid-star-fixed" style="font-size:16px;top:8%;left:55%;animation-duration:2.8s;animation-delay:1s;">\u2b50</div>
    <div class="kid-star-fixed" style="font-size:12px;top:15%;left:80%;animation-duration:3.5s;animation-delay:0.3s;">\u2b50</div>
    <div class="kid-star-fixed" style="font-size:15px;top:35%;left:5%;animation-duration:4s;animation-delay:0.8s;">\u2b50</div>
    <div class="kid-star-fixed" style="font-size:11px;top:50%;left:92%;animation-duration:3.2s;animation-delay:1.5s;">\u2b50</div>
    <div class="kid-star-fixed" style="font-size:17px;top:65%;left:15%;animation-duration:2.6s;animation-delay:0.2s;">\u2b50</div>
    <div class="kid-star-fixed" style="font-size:13px;top:75%;left:70%;animation-duration:3.8s;animation-delay:2s;">\u2b50</div>
    <div class="kid-star-fixed" style="font-size:10px;top:88%;left:45%;animation-duration:3s;animation-delay:1.2s;">\u2b50</div>
    <div class="kid-star-fixed" style="font-size:14px;top:42%;left:88%;animation-duration:2.4s;animation-delay:0.7s;">\u2b50</div>
    <!-- Clouds -->
    <div class="kid-cloud" style="top:3%;animation-duration:40s;"></div>
    <div class="kid-cloud" style="top:20%;animation-duration:55s;animation-delay:12s;width:90px;height:38px;"></div>
    <div class="kid-cloud" style="top:45%;animation-duration:48s;animation-delay:25s;width:100px;height:42px;"></div>
    <div class="kid-cloud" style="top:65%;animation-duration:52s;animation-delay:8s;width:80px;height:34px;"></div>
    <div class="kid-cloud" style="top:85%;animation-duration:44s;animation-delay:20s;"></div>
</div>
<!-- Corner characters: cute SVG creatures -->
<!-- Cat (bottom-left) -->
<div style="position:fixed;bottom:15px;left:260px;z-index:2;opacity:0.4;pointer-events:none;animation:kid-bob 3s ease-in-out infinite;">
<svg width="55" height="50" viewBox="0 0 55 50">
    <polygon points="9,16 4,2 17,12" fill="#f97316"/>
    <polygon points="46,16 51,2 39,12" fill="#f97316"/>
    <ellipse cx="27" cy="27" rx="20" ry="16" fill="#fb923c"/>
    <ellipse cx="20" cy="25" rx="2.5" ry="3" fill="#1e293b"/>
    <ellipse cx="34" cy="25" rx="2.5" ry="3" fill="#1e293b"/>
    <circle cx="21" cy="24" r="0.8" fill="white"/>
    <circle cx="35" cy="24" r="0.8" fill="white"/>
    <ellipse cx="27" cy="30" rx="2" ry="1.2" fill="#f472b6"/>
    <path d="M24 32 Q27 35 30 32" stroke="#ea580c" stroke-width="1" fill="none"/>
    <line x1="7" y1="27" x2="0" y2="25" stroke="#fdba74" stroke-width="1.2"/>
    <line x1="7" y1="29" x2="0" y2="30" stroke="#fdba74" stroke-width="1.2"/>
    <line x1="47" y1="27" x2="55" y2="25" stroke="#fdba74" stroke-width="1.2"/>
    <line x1="47" y1="29" x2="55" y2="30" stroke="#fdba74" stroke-width="1.2"/>
    <path d="M13 43 Q7 47 10 50" stroke="#fb923c" stroke-width="3.5" fill="none" stroke-linecap="round"/>
</svg></div>
<!-- Dog (bottom-right) -->
<div style="position:fixed;bottom:15px;right:25px;z-index:2;opacity:0.4;pointer-events:none;animation:kid-bob 3.5s ease-in-out 0.5s infinite;">
<svg width="55" height="50" viewBox="0 0 55 50">
    <ellipse cx="10" cy="10" rx="9" ry="13" fill="#a16207" transform="rotate(-20,10,10)"/>
    <ellipse cx="45" cy="10" rx="9" ry="13" fill="#a16207" transform="rotate(20,45,10)"/>
    <circle cx="27" cy="25" r="18" fill="#d97706"/>
    <ellipse cx="20" cy="22" rx="2.5" ry="3" fill="#1e293b"/>
    <ellipse cx="34" cy="22" rx="2.5" ry="3" fill="#1e293b"/>
    <circle cx="21" cy="21" r="0.8" fill="white"/>
    <circle cx="35" cy="21" r="0.8" fill="white"/>
    <ellipse cx="27" cy="29" rx="3.5" ry="2.5" fill="#1e293b"/>
    <ellipse cx="27" cy="28" rx="2" ry="1.2" fill="#f472b6"/>
    <path d="M22 33 Q27 38 32 33" stroke="#92400e" stroke-width="1.2" fill="none"/>
</svg></div>
<!-- Unicorn (top-right) -->
<div style="position:fixed;top:75px;right:25px;z-index:2;opacity:0.35;pointer-events:none;animation:kid-bob 4s ease-in-out 1s infinite;">
<svg width="50" height="55" viewBox="0 0 50 55">
    <polygon points="25,0 22,15 28,15" fill="#fbbf24"/>
    <circle cx="25" cy="25" r="14" fill="white" stroke="#e9d5ff" stroke-width="1"/>
    <ellipse cx="19" cy="23" rx="2.5" ry="3" fill="#1e293b"/>
    <ellipse cx="31" cy="23" rx="2.5" ry="3" fill="#1e293b"/>
    <circle cx="20" cy="22" r="0.8" fill="white"/>
    <circle cx="32" cy="22" r="0.8" fill="white"/>
    <circle cx="14" cy="28" rx="3" fill="#fecdd3" opacity="0.5"/>
    <circle cx="36" cy="28" rx="3" fill="#fecdd3" opacity="0.5"/>
    <path d="M20 30 Q25 34 30 30" stroke="#ec4899" stroke-width="1.2" fill="none"/>
    <path d="M11 16 Q5 10 7 18" stroke="#c4b5fd" stroke-width="2.5" fill="none" stroke-linecap="round"/>
    <path d="M13 14 Q8 6 9 15" stroke="#fbcfe8" stroke-width="2" fill="none" stroke-linecap="round"/>
    <path d="M39 16 Q45 10 43 18" stroke="#bfdbfe" stroke-width="2.5" fill="none" stroke-linecap="round"/>
    <path d="M37 14 Q42 6 41 15" stroke="#fde68a" stroke-width="2" fill="none" stroke-linecap="round"/>
</svg></div>
<!-- Rocket (top-left past sidebar) -->
<div style="position:fixed;top:65px;left:260px;z-index:2;opacity:0.35;pointer-events:none;animation:kid-bob 3.2s ease-in-out 0.8s infinite;">
<svg width="35" height="55" viewBox="0 0 35 55">
    <ellipse cx="17" cy="22" rx="10" ry="18" fill="#ef4444"/>
    <ellipse cx="17" cy="22" rx="6.5" ry="12" fill="#fca5a5"/>
    <circle cx="17" cy="19" r="4.5" fill="#dbeafe"/>
    <circle cx="17" cy="19" r="2.5" fill="#3b82f6"/>
    <polygon points="17,1 14,10 20,10" fill="#ef4444"/>
    <polygon points="7,34 2,43 12,36" fill="#f97316"/>
    <polygon points="27,34 32,43 22,36" fill="#f97316"/>
    <ellipse cx="17" cy="40" rx="4" ry="3.5" fill="#fbbf24"/>
    <ellipse cx="17" cy="44" rx="2.5" ry="5" fill="#fb923c" opacity="0.7"/>
    <ellipse cx="17" cy="49" rx="1.5" ry="3.5" fill="#fbbf24" opacity="0.4"/>
</svg></div>
<!-- SVG Mascots -->
<div class="kid-mascot-row">
    <div class="kid-mascot">
        <div class="kid-speech">Ich schreibe!</div>
        <svg width="70" height="75" viewBox="0 0 70 75">
            <!-- Textino: cute blue robot -->
            <!-- Antenna -->
            <line x1="35" y1="8" x2="35" y2="0" stroke="#60a5fa" stroke-width="2.5" stroke-linecap="round"/>
            <circle cx="35" cy="0" r="4" fill="#fbbf24"/>
            <!-- Head -->
            <rect x="10" y="8" width="50" height="32" rx="12" fill="#3b82f6"/>
            <!-- Face screen -->
            <rect x="15" y="13" width="40" height="22" rx="8" fill="#dbeafe"/>
            <!-- Eyes -->
            <circle cx="27" cy="23" r="5" fill="white"/>
            <circle cx="43" cy="23" r="5" fill="white"/>
            <circle cx="28" cy="23" r="3" fill="#1e293b"/>
            <circle cx="44" cy="23" r="3" fill="#1e293b"/>
            <!-- Eye shine -->
            <circle cx="29" cy="22" r="1" fill="white"/>
            <circle cx="45" cy="22" r="1" fill="white"/>
            <!-- Smile -->
            <path d="M25 29 Q35 35 45 29" stroke="#3b82f6" stroke-width="2" fill="none" stroke-linecap="round"/>
            <!-- Body -->
            <rect x="18" y="40" width="34" height="22" rx="8" fill="#60a5fa"/>
            <!-- Arms -->
            <rect x="5" y="42" width="13" height="8" rx="4" fill="#93c5fd"/>
            <rect x="52" y="42" width="13" height="8" rx="4" fill="#93c5fd"/>
            <!-- Pencil in right hand -->
            <line x1="65" y1="42" x2="69" y2="32" stroke="#f97316" stroke-width="3" stroke-linecap="round"/>
            <polygon points="69,32 67,28 71,28" fill="#fbbf24"/>
            <!-- Belly button -->
            <circle cx="35" cy="51" r="3" fill="#3b82f6"/>
            <!-- Feet -->
            <rect x="20" y="62" width="12" height="8" rx="4" fill="#3b82f6"/>
            <rect x="38" y="62" width="12" height="8" rx="4" fill="#3b82f6"/>
        </svg>
        <div class="kid-mascot-name">Textino</div>
    </div>
    <div class="kid-mascot">
        <div class="kid-speech">Ich male!</div>
        <svg width="70" height="75" viewBox="0 0 70 75">
            <!-- Pixela: cute pink artist character -->
            <!-- Beret -->
            <ellipse cx="35" cy="10" rx="22" ry="8" fill="#ec4899"/>
            <circle cx="35" cy="5" r="5" fill="#f472b6"/>
            <!-- Head -->
            <circle cx="35" cy="25" r="20" fill="#fda4af"/>
            <!-- Rosy cheeks -->
            <circle cx="22" cy="29" r="5" fill="#fecdd3" opacity="0.7"/>
            <circle cx="48" cy="29" r="5" fill="#fecdd3" opacity="0.7"/>
            <!-- Eyes -->
            <ellipse cx="27" cy="23" rx="4.5" ry="5" fill="white"/>
            <ellipse cx="43" cy="23" rx="4.5" ry="5" fill="white"/>
            <circle cx="28" cy="23" r="3" fill="#1e293b"/>
            <circle cx="44" cy="23" r="3" fill="#1e293b"/>
            <circle cx="29" cy="22" r="1" fill="white"/>
            <circle cx="45" cy="22" r="1" fill="white"/>
            <!-- Cat mouth -->
            <path d="M30 31 L35 34 L40 31" stroke="#e11d48" stroke-width="1.5" fill="none" stroke-linecap="round"/>
            <!-- Body -->
            <rect x="20" y="45" width="30" height="18" rx="10" fill="#fb7185"/>
            <!-- Arms -->
            <rect x="7" y="47" width="13" height="7" rx="3.5" fill="#fda4af"/>
            <rect x="50" y="47" width="13" height="7" rx="3.5" fill="#fda4af"/>
            <!-- Paintbrush in right hand -->
            <line x1="63" y1="47" x2="68" y2="35" stroke="#a16207" stroke-width="2.5" stroke-linecap="round"/>
            <ellipse cx="68" cy="33" rx="4" ry="5" fill="#8b5cf6" transform="rotate(-15,68,33)"/>
            <!-- Paint palette in left hand -->
            <ellipse cx="4" cy="50" rx="8" ry="5" fill="#fde68a" transform="rotate(10,4,50)"/>
            <circle cx="2" cy="48" r="2" fill="#ef4444"/>
            <circle cx="6" cy="47" r="2" fill="#3b82f6"/>
            <circle cx="4" cy="52" r="2" fill="#22c55e"/>
            <!-- Feet -->
            <ellipse cx="28" cy="67" rx="7" ry="5" fill="#ec4899"/>
            <ellipse cx="42" cy="67" rx="7" ry="5" fill="#ec4899"/>
        </svg>
        <div class="kid-mascot-name">Pixela</div>
    </div>
    <div class="kid-mascot">
        <div class="kid-speech">Ich spiele!</div>
        <svg width="70" height="75" viewBox="0 0 70 75">
            <!-- Soundo: cute orange music character -->
            <!-- Headphones band -->
            <path d="M12 25 Q12 5 35 5 Q58 5 58 25" stroke="#f97316" stroke-width="4" fill="none" stroke-linecap="round"/>
            <!-- Headphone pads -->
            <rect x="6" y="20" width="12" height="16" rx="6" fill="#f97316"/>
            <rect x="52" y="20" width="12" height="16" rx="6" fill="#f97316"/>
            <rect x="8" y="22" width="8" height="12" rx="4" fill="#fdba74"/>
            <rect x="54" y="22" width="8" height="12" rx="4" fill="#fdba74"/>
            <!-- Head -->
            <circle cx="35" cy="28" r="18" fill="#fed7aa"/>
            <!-- Eyes - happy closed -->
            <path d="M24 26 Q28 22 32 26" stroke="#1e293b" stroke-width="2.5" fill="none" stroke-linecap="round"/>
            <path d="M38 26 Q42 22 46 26" stroke="#1e293b" stroke-width="2.5" fill="none" stroke-linecap="round"/>
            <!-- Big open smile -->
            <path d="M25 33 Q35 42 45 33" stroke="#ea580c" stroke-width="2" fill="#fef3c7" stroke-linecap="round"/>
            <!-- Body -->
            <rect x="22" y="46" width="26" height="16" rx="8" fill="#fb923c"/>
            <!-- Arms -->
            <rect x="9" y="48" width="13" height="7" rx="3.5" fill="#fdba74"/>
            <rect x="48" y="48" width="13" height="7" rx="3.5" fill="#fdba74"/>
            <!-- Music notes floating -->
            <text x="60" y="15" font-size="14" fill="#8b5cf6" opacity="0.8">\u266a</text>
            <text x="4" y="12" font-size="11" fill="#ec4899" opacity="0.7">\u266b</text>
            <text x="55" y="45" font-size="10" fill="#f97316" opacity="0.6">\u266a</text>
            <!-- Feet -->
            <ellipse cx="29" cy="66" rx="7" ry="5" fill="#f97316"/>
            <ellipse cx="41" cy="66" rx="7" ry="5" fill="#f97316"/>
        </svg>
        <div class="kid-mascot-name">Soundo</div>
    </div>
</div>
"""


def _kid_stars(v: Optional[float]) -> str:
    """Convert a 0-1 score to 1-5 star rating HTML."""
    if v is None:
        return "\u2b50" * 0
    n = max(1, min(5, round(v * 10)))  # 0.1→1 star, 0.5→5 stars
    return "\u2b50" * n + "\u2606" * (5 - n)  # filled + empty


def _kid_emoji(v: Optional[float]) -> str:
    """Return emoji face based on coherence score."""
    if v is None:
        return "\U0001f914"
    if v >= 0.45:
        return "\U0001f929"  # star-struck
    if v >= 0.35:
        return "\U0001f60a"  # happy
    if v >= 0.25:
        return "\U0001f642"  # slightly smiling
    return "\U0001f61f"  # worried


def _kid_verdict(v: Optional[float], lang: str = "de") -> str:
    """Return kid-friendly verdict text."""
    if v is None:
        return "Hmm..." if lang == "de" else "Hmm..."
    if lang == "de":
        if v >= 0.45:
            return "Super! Alles passt perfekt zusammen! \U0001f389"
        if v >= 0.35:
            return "Gut gemacht! Das passt ziemlich gut! \U0001f44d"
        if v >= 0.25:
            return "Geht so \u2014 ein bisschen passt es! \U0001f914"
        return "Hmm, das passt noch nicht so gut \U0001f61e"
    else:
        if v >= 0.45:
            return "Amazing! Everything fits perfectly together! \U0001f389"
        if v >= 0.35:
            return "Well done! That fits pretty well! \U0001f44d"
        if v >= 0.25:
            return "So-so \u2014 it fits a little bit! \U0001f914"
        return "Hmm, that doesn't quite fit yet \U0001f61e"


def kid_score_card(label: str, value: Optional[float], is_main: bool = False) -> str:
    """Kid-friendly score card with stars and emoji."""
    cls = "kid-sc-main" if is_main else (
        "kid-sc-great" if value and value >= 0.45 else
        "kid-sc-ok" if value and value >= 0.30 else "kid-sc-low"
    )
    stars = _kid_stars(value)
    emoji = _kid_emoji(value) if is_main else ""
    val_str = f"{value:.3f}" if value is not None else "\u2014"
    emoji_html = f'<div class="kid-sc-emoji">{emoji}</div>' if emoji else ""
    return (
        f'<div class="kid-sc {cls} kid-confetti">'
        f'<div class="kid-sc-lbl">{label}</div>'
        f'{emoji_html}'
        f'<div class="kid-sc-stars">{stars}</div>'
        f'<div class="kid-sc-val">{val_str}</div>'
        f'</div>'
    )


# Kid-mode UI labels
UI_LABELS_KID = {
    "de": {
        "hero_title": "Multimodale KI f\u00fcr Kids",
        "hero_sub": "Beschreibe eine Szene und die KI erzeugt <b>Text + Bild + Audio</b> dazu!",
        "config": "Einstellungen",
        "backend": "Wie soll es erstellt werden?",
        "planning": "Planungsmodus",
        "language": "Sprache",
        "examples": "Ideen zum Ausprobieren",
        "scene_placeholder": "Beschreibe deine Szene hier... z.B. 'Ein Einhorn fliegt \u00fcber einen Regenbogen' \U0001f308",
        "generate_btn": "\u2728 Los geht's!",
        "welcome_text": "Beschreibe eine Szene und klicke auf <b>\u2728 Los geht's!</b>",
        "welcome_hint": "oder w\u00e4hle eine Idee aus der Seitenleiste \U0001f449",
        "scores_label": "\U0001f3af Wie gut passt alles zusammen?",
        "gen_text_label": "\U0001f916 Textino schreibt...",
        "gen_image_label": "\U0001f3a8 Pixela malt...",
        "gen_audio_label": "\U0001f3b5 Soundo spielt...",
        "translated_note": "Aus dem Deutschen \u00fcbersetzt",
        "original_label": "Original (Deutsch)",
    },
    "en": {
        "hero_title": "Multimodal AI for Kids",
        "hero_sub": "Describe a scene and the AI creates <b>text + image + audio</b> for it!",
        "config": "Settings",
        "backend": "How should it be created?",
        "planning": "Planning Mode",
        "language": "Language",
        "examples": "Ideas to Try",
        "scene_placeholder": "Describe your scene here... e.g., 'A unicorn flying over a rainbow' \U0001f308",
        "generate_btn": "\u2728 Let's Go!",
        "welcome_text": "Describe a scene and click <b>\u2728 Let's Go!</b>",
        "welcome_hint": "or pick an idea from the sidebar \U0001f449",
        "scores_label": "\U0001f3af How well does everything fit together?",
        "gen_text_label": "\U0001f916 Textino writes...",
        "gen_image_label": "\U0001f3a8 Pixela paints...",
        "gen_audio_label": "\U0001f3b5 Soundo plays...",
        "translated_note": "Translated from German",
        "original_label": "Original (German)",
    },
}

# ---------------------------------------------------------------------------
# Planning prompt template (same as src/planner/prompts/unified.txt)
# ---------------------------------------------------------------------------
PLAN_PROMPT_TEMPLATE = """You must produce a SINGLE valid JSON object.

RULES:
- Every field MUST exist
- Fields that represent lists MUST be arrays
- Strings must never be arrays
- Use short phrases, not long paragraphs
- Do NOT include explanations
- Do NOT include markdown
- Do NOT truncate

Schema:
{
  "scene_summary": string,
  "domain": string,

  "core_semantics": {
    "setting": string,
    "time_of_day": string,
    "weather": string,
    "main_subjects": [string],
    "actions": [string]
  },

  "style_controls": {
    "visual_style": [string],
    "color_palette": [string],
    "lighting": [string],
    "camera": [string],
    "mood_emotion": [string],
    "narrative_tone": [string]
  },

  "image_constraints": {
    "must_include": [string],
    "must_avoid": [string],
    "objects": [string],
    "environment_details": [string],
    "composition": [string]
  },

  "audio_constraints": {
    "audio_intent": [string],
    "sound_sources": [string],
    "ambience": [string],
    "tempo": string,
    "must_include": [string],
    "must_avoid": [string]
  },

  "text_constraints": {
    "must_include": [string],
    "must_avoid": [string],
    "keywords": [string],
    "length": string
  }
}

User request:
"""

EXTENDED_PLAN_SYSTEM = """You are an expert multimodal content planner. Create a detailed,
comprehensive semantic plan for generating coherent multimodal content (text, image, audio).

You have an extended budget. Take your time to:
1. Deeply analyze the user's request
2. Consider multiple perspectives and interpretations
3. Ensure semantic consistency across all modalities
4. Provide rich, detailed specifications

Think step by step about what visual elements, sounds, and descriptive text would best represent the scene.
After your analysis, produce a SINGLE valid JSON object matching the schema."""


# ---------------------------------------------------------------------------
# Cached model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_coherence_engine():
    from src.coherence.coherence_engine import CoherenceEngine
    return CoherenceEngine(target_dim=512)

@st.cache_resource
def load_image_retriever():
    from src.generators.image.generator_improved import ImprovedImageRetrievalGenerator
    return ImprovedImageRetrievalGenerator(index_path="data/embeddings/image_index.npz", min_similarity=0.20)

@st.cache_resource
def load_audio_retriever():
    from src.generators.audio.retrieval import AudioRetrievalGenerator
    return AudioRetrievalGenerator(index_path="data/embeddings/audio_index.npz", min_similarity=0.10)

@st.cache_resource
def get_inference_client():
    """Default client for text generation (auto-routes to available providers)."""
    from huggingface_hub import InferenceClient
    token = os.environ.get("HF_TOKEN")
    return InferenceClient(token=token)



# ---------------------------------------------------------------------------
# Translation (German <-> English)
# ---------------------------------------------------------------------------

TRANSLATION_MODELS = {
    "de-en": "Helsinki-NLP/opus-mt-de-en",
    "en-de": "Helsinki-NLP/opus-mt-en-de",
}


def translate(text: str, direction: str) -> str:
    """Translate text using HF Inference API. direction: 'de-en' or 'en-de'."""
    if not text or not text.strip():
        return text
    model_id = TRANSLATION_MODELS[direction]
    client = get_inference_client()
    try:
        result = client.translation(text, model=model_id)
        if isinstance(result, str):
            return result
        # huggingface_hub returns a TranslationOutput object
        return result.translation_text if hasattr(result, "translation_text") else str(result)
    except Exception as e:
        logger.warning("Translation (%s) failed: %s — returning original", direction, e)
        return text


def translate_de_to_en(text: str) -> str:
    return translate(text, "de-en")


def translate_en_to_de(text: str) -> str:
    return translate(text, "en-de")


# ---------------------------------------------------------------------------
# UI labels (i18n)
# ---------------------------------------------------------------------------

UI_LABELS = {
    "en": {
        "hero_title": "Multimodal Coherence AI",
        "hero_sub": 'Generate semantically coherent <b>text + image + audio</b> bundles '
                    'and evaluate cross-modal alignment with the <b>MSCI</b> metric.',
        "config": "Configuration",
        "backend": "Backend",
        "planning": "Planning Mode",
        "language": "Language",
        "examples": "Examples",
        "scene_placeholder": "Describe a scene... e.g., 'A peaceful forest at dawn with birdsong and morning mist'",
        "generate_btn": "Generate Bundle",
        "welcome_text": 'Enter a scene description and click <b>Generate Bundle</b>',
        "welcome_hint": "or pick an example from the sidebar",
        "scores_label": "Coherence Scores",
        "gen_text_label": "Generated Text",
        "gen_image_label": "Generated Image",
        "gen_audio_label": "Generated Audio",
        "translated_note": "Translated from German",
        "original_label": "Original (German)",
    },
    "de": {
        "hero_title": "Multimodale Koh\u00e4renz-KI",
        "hero_sub": 'Erzeuge semantisch koh\u00e4rente <b>Text + Bild + Audio</b> B\u00fcndel '
                    'und bewerte die modale \u00dcbereinstimmung mit der <b>MSCI</b>-Metrik.',
        "config": "Einstellungen",
        "backend": "Verfahren",
        "planning": "Planungsmodus",
        "language": "Sprache",
        "examples": "Beispiele",
        "scene_placeholder": "Beschreibe eine Szene... z.B. 'Ein friedlicher Wald bei Sonnenaufgang mit Vogelgesang'",
        "generate_btn": "B\u00fcndel erzeugen",
        "welcome_text": 'Beschreibe eine Szene und klicke auf <b>B\u00fcndel erzeugen</b>',
        "welcome_hint": "oder w\u00e4hle ein Beispiel aus der Seitenleiste",
        "scores_label": "Koh\u00e4renz-Bewertung",
        "gen_text_label": "Erzeugter Text",
        "gen_image_label": "Erzeugtes Bild",
        "gen_audio_label": "Erzeugtes Audio",
        "translated_note": "Aus dem Deutschen \u00fcbersetzt",
        "original_label": "Original (Deutsch)",
    },
}


# ---------------------------------------------------------------------------
# LLM helpers — Groq (primary) + Pollinations (fallback)
# ---------------------------------------------------------------------------

import requests as _requests
import urllib.parse as _urlparse

# Groq models — free tier, fast LPU inference
GROQ_MODELS = [
    "llama-3.3-70b-versatile",     # 30 RPM, 1K RPD
    "llama-3.1-8b-instant",        # 30 RPM, 14.4K RPD (fast fallback)
]
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

# Pollinations — free, no auth, OpenAI-compatible
POLLINATIONS_TEXT_URL = "https://text.pollinations.ai/openai"


def _groq_chat(system: str, user: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
    """Call Groq API (OpenAI-compatible). Tries multiple models."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    last_error = None
    for model_id in GROQ_MODELS:
        try:
            resp = _requests.post(
                GROQ_BASE_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                text = resp.json()["choices"][0]["message"]["content"].strip()
                if text:
                    return text
            elif resp.status_code == 429:
                logger.warning("Groq %s rate-limited (429), trying next", model_id)
            else:
                logger.warning("Groq %s returned %s: %s", model_id, resp.status_code, resp.text[:200])
        except Exception as e:
            last_error = e
            logger.warning("Groq %s failed: %s", model_id, e)
        continue
    raise RuntimeError(f"All Groq models failed. Last: {last_error}")


def _pollinations_chat(system: str, user: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
    """Call Pollinations text API (free, no auth, OpenAI-compatible)."""
    resp = _requests.post(
        POLLINATIONS_TEXT_URL,
        json={
            "model": "openai",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=60,
    )
    if resp.status_code == 200:
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if text:
            return text
    raise RuntimeError(f"Pollinations text failed: {resp.status_code}")


def _llm_chat(system: str, user: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
    """Unified LLM chat: Groq → Pollinations fallback."""
    # Try Groq first (fast, high quality)
    try:
        return _groq_chat(system, user, max_tokens, temperature)
    except Exception as e:
        logger.warning("Groq failed: %s — trying Pollinations", e)

    # Fallback to Pollinations (free, no auth)
    return _pollinations_chat(system, user, max_tokens, temperature)


def _parse_plan_json(raw: str) -> Optional[Dict[str, Any]]:
    """Parse a semantic plan JSON from LLM output, with repair."""
    from src.utils.json_repair import try_repair_json
    return try_repair_json(raw)


def _validate_and_build_plan(data: Dict[str, Any]):
    """Validate and build a SemanticPlan from dict."""
    from src.planner.validation import validate_semantic_plan_dict
    from src.planner.schema import SemanticPlan
    validate_semantic_plan_dict(data)
    return SemanticPlan(**data)


# ---------------------------------------------------------------------------
# Planning functions (Groq / Pollinations)
# ---------------------------------------------------------------------------

def plan_single(prompt: str) -> Optional[Any]:
    """Single planner call via HF API. Returns SemanticPlan or None."""
    system = "You are a multimodal content planner. Output ONLY valid JSON, no explanations."
    user = PLAN_PROMPT_TEMPLATE + prompt
    try:
        raw = _llm_chat(system, user, max_tokens=1200, temperature=0.3)
        data = _parse_plan_json(raw)
        if data:
            return _validate_and_build_plan(data)
    except Exception as e:
        logger.warning("Planner call failed: %s", e)
    return None


def plan_council(prompt: str) -> Optional[Any]:
    """Council mode: 3 planner calls merged. Returns SemanticPlan or None."""
    plans = []
    temps = [0.2, 0.4, 0.5]  # Slightly different temperatures for diversity
    system = "You are a multimodal content planner. Output ONLY valid JSON, no explanations."
    user = PLAN_PROMPT_TEMPLATE + prompt

    for temp in temps:
        try:
            raw = _llm_chat(system, user, max_tokens=1200, temperature=temp)
            data = _parse_plan_json(raw)
            if data:
                plan = _validate_and_build_plan(data)
                plans.append(plan)
        except Exception as e:
            logger.warning("Council call failed (temp=%.1f): %s", temp, e)

    if not plans:
        return None
    if len(plans) == 1:
        return plans[0]

    # Merge using existing merge logic
    try:
        from src.planner.merge_logic import merge_council_plans
        while len(plans) < 3:
            plans.append(plans[0])  # Pad if fewer than 3
        merged, _ = merge_council_plans(plans[0], plans[1], plans[2])
        return merged
    except Exception as e:
        logger.warning("Merge failed: %s — using first plan", e)
        return plans[0]


def plan_extended(prompt: str) -> Optional[Any]:
    """Extended prompt mode: longer system prompt, more tokens. Returns SemanticPlan or None."""
    user = PLAN_PROMPT_TEMPLATE + prompt
    try:
        raw = _llm_chat(EXTENDED_PLAN_SYSTEM, user, max_tokens=2000, temperature=0.35)
        data = _parse_plan_json(raw)
        if data:
            return _validate_and_build_plan(data)
    except Exception as e:
        logger.warning("Extended planner failed: %s", e)
    return None


# ---------------------------------------------------------------------------
# Generation / retrieval functions
# ---------------------------------------------------------------------------

# Pollinations endpoints
POLLINATIONS_IMAGE_FREE_URL = "https://image.pollinations.ai/prompt"  # Free, no auth
POLLINATIONS_GEN_IMAGE_URL = "https://gen.pollinations.ai/image"       # Needs API key
POLLINATIONS_AUDIO_URL = "https://gen.pollinations.ai/v1/audio/speech"  # Needs API key (TTS only)
POLLINATIONS_TTS_URL = "https://gen.pollinations.ai/audio"              # Needs API key (TTS only)

# ElevenLabs (sound effects — actual ambient sounds, NOT speech)
ELEVENLABS_SFX_URL = "https://api.elevenlabs.io/v1/sound-generation"

# Stable Horde (free, crowdsourced, no key)
STABLE_HORDE_URL = "https://stablehorde.net/api/v2"


def _pollinations_headers() -> dict:
    """Get auth headers for Pollinations gen.pollinations.ai endpoints."""
    key = os.environ.get("POLLINATIONS_API_KEY", "")
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


def gen_text(prompt: str, mode: str) -> dict:
    """Generate text and optional plan using Groq / Pollinations."""
    # Step 1: Plan (if not direct mode)
    plan = None
    image_prompt = prompt
    audio_prompt = prompt

    if mode == "planner":
        plan = plan_single(prompt)
    elif mode == "council":
        plan = plan_council(prompt)
    elif mode == "extended_prompt":
        plan = plan_extended(prompt)

    # Extract modality-specific prompts from plan
    if plan is not None:
        try:
            from src.planner.schema_to_text import plan_to_prompts
            prompts = plan_to_prompts(plan)
            image_prompt = prompts["image_prompt"]
            audio_prompt = prompts["audio_prompt"]
            text_input = prompts["text_prompt"]
        except Exception as e:
            logger.warning("plan_to_prompts failed: %s", e)
            text_input = prompt
    else:
        text_input = prompt

    # Step 2: Generate text via Groq / Pollinations
    system_prompt = (
        "You are a concise descriptive writer. "
        "Write a literal description of the scene in 3 to 5 natural sentences. "
        "No bullet points, no numbered lists, no meta commentary. "
        "Focus on concrete visual details AND the likely audio ambience."
    )
    text_error = None
    try:
        text = _llm_chat(system_prompt, f"Describe this scene: {text_input}", max_tokens=250, temperature=0.7)
        if not text:
            raise ValueError("Empty response")
    except Exception as e:
        logger.warning("Text gen failed: %s — using prompt", e)
        text = prompt
        text_error = str(e)

    return {
        "text": text,
        "image_prompt": image_prompt,
        "audio_prompt": audio_prompt,
        "plan": plan.model_dump() if plan and hasattr(plan, "model_dump") else None,
        "text_error": text_error,
    }


def _stable_horde_image(prompt: str, timeout: int = 90) -> Optional[bytes]:
    """Generate image via Stable Horde (free, crowdsourced, no API key needed).

    Submits an async job, polls for completion, downloads the result.
    Returns image bytes or None on failure.
    """
    # Submit job (anonymous key)
    try:
        submit = _requests.post(
            "https://stablehorde.net/api/v2/generate/async",
            json={
                "prompt": f"{prompt}, high quality, detailed, digital art",
                "params": {"width": 768, "height": 768, "steps": 25},
                "nsfw": False,
                "models": ["FLUX.1 [schnell]"],
            },
            headers={"apikey": "0000000000"},
            timeout=15,
        )
        if submit.status_code != 202:
            logger.warning("Stable Horde submit: %s %s", submit.status_code, submit.text[:200])
            return None
        job_id = submit.json().get("id")
        if not job_id:
            return None
    except Exception as e:
        logger.warning("Stable Horde submit failed: %s", e)
        return None

    # Poll for completion
    import time as _time
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        _time.sleep(3)
        try:
            check = _requests.get(
                f"https://stablehorde.net/api/v2/generate/check/{job_id}", timeout=10,
            )
            status = check.json()
            if status.get("done"):
                # Fetch result
                result = _requests.get(
                    f"https://stablehorde.net/api/v2/generate/status/{job_id}", timeout=10,
                )
                gens = result.json().get("generations", [])
                if gens:
                    img_url = gens[0].get("img", "")
                    if img_url.startswith("http"):
                        img_resp = _requests.get(img_url, timeout=30)
                        if img_resp.status_code == 200 and len(img_resp.content) > 1000:
                            return img_resp.content
                return None
            if status.get("faulted"):
                logger.warning("Stable Horde job faulted")
                return None
        except Exception as e:
            logger.warning("Stable Horde poll error: %s", e)
    logger.warning("Stable Horde timed out after %ds", timeout)
    return None


def generate_image(prompt: str) -> dict:
    """Generate image: Pollinations (auth) → Pollinations (free) → Stable Horde → CLIP retrieval."""
    # --- Attempt 1: Pollinations gen.pollinations.ai (with API key) ---
    headers = _pollinations_headers()
    if headers:
        try:
            encoded = _urlparse.quote(prompt)
            url = f"{POLLINATIONS_GEN_IMAGE_URL}/{encoded}?model=flux&width=1024&height=1024&nologo=true"
            resp = _requests.get(url, headers=headers, timeout=60)
            if resp.status_code == 200 and len(resp.content) > 1000:
                ct = resp.headers.get("content-type", "")
                suffix = ".jpg" if "jpeg" in ct else ".png"
                tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir="/tmp")
                tmp.write(resp.content)
                tmp.flush()
                return {
                    "path": tmp.name, "backend": "generative",
                    "model": "Pollinations-FLUX", "failed": False,
                }
            logger.warning("Pollinations auth image returned %s", resp.status_code)
        except Exception as e:
            logger.warning("Pollinations auth image failed: %s", e)

    # --- Attempt 2: Pollinations free endpoint (image.pollinations.ai, no auth) ---
    try:
        encoded = _urlparse.quote(prompt)
        url = f"{POLLINATIONS_IMAGE_FREE_URL}/{encoded}?model=flux&width=1024&height=1024&nologo=true"
        resp = _requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 1000:
            ct = resp.headers.get("content-type", "")
            suffix = ".jpg" if "jpeg" in ct else ".png"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir="/tmp")
            tmp.write(resp.content)
            tmp.flush()
            return {
                "path": tmp.name, "backend": "generative",
                "model": "Pollinations-FLUX", "failed": False,
            }
        logger.warning("Pollinations free image returned %s", resp.status_code)
    except Exception as e:
        logger.warning("Pollinations free image failed: %s", e)

    # --- Attempt 3: Stable Horde (free, crowdsourced, ~30-40s) ---
    try:
        img_bytes = _stable_horde_image(prompt)
        if img_bytes:
            tmp = tempfile.NamedTemporaryFile(suffix=".webp", delete=False, dir="/tmp")
            tmp.write(img_bytes)
            tmp.flush()
            return {
                "path": tmp.name, "backend": "generative",
                "model": "StableHorde-FLUX", "failed": False,
            }
    except Exception as e:
        logger.warning("Stable Horde failed: %s", e)

    # --- Fallback: CLIP retrieval ---
    logger.info("All image gen failed — using CLIP retrieval")
    return retrieve_image(prompt)


def _make_audio_query(scene_prompt: str) -> str:
    """Use LLM to convert a scene description into an audio-focused search query."""
    try:
        result = _llm_chat(
            system=(
                "Convert the scene into a short ambient sound description (max 15 words). "
                "Describe ONLY the sounds you would hear — no visuals, no story. "
                "Examples: 'gentle rain on leaves with distant thunder', "
                "'busy city traffic with car horns and pedestrians', "
                "'ocean waves on sandy beach with seagulls calling'."
            ),
            user=scene_prompt,
            max_tokens=60,
            temperature=0.3,
        )
        query = result.strip().strip('"').strip("'")
        if len(query) > 10:
            logger.info("Audio query: %s -> %s", scene_prompt[:50], query)
            return query
    except Exception as e:
        logger.warning("Audio query LLM failed: %s", e)
    return scene_prompt


def _stable_audio_generate(prompt: str, duration: float = 8.0) -> Optional[str]:
    """Generate ambient audio via Stable Audio Open (free Gradio Space, no API key).

    Returns path to generated WAV file or None on failure.
    """
    try:
        from gradio_client import Client as GradioClient
        client = GradioClient("artificialguybr/Stable-Audio-Open-Zero", verbose=False)
        result = client.predict(
            prompt=prompt,
            seconds_total=duration,
            steps=50,
            cfg_scale=7,
            api_name="/predict",
        )
        if result and os.path.exists(result):
            logger.info("Stable Audio generated: %s (%d bytes)", result, os.path.getsize(result))
            return result
        logger.warning("Stable Audio returned invalid path: %s", result)
    except Exception as e:
        logger.warning("Stable Audio failed: %s", e)
    return None


def generate_audio(prompt: str) -> dict:
    """Generate ambient audio via Stable Audio Open → AI-enhanced CLAP retrieval.

    1. LLM converts scene prompt into a sound-focused query
    2. Stable Audio Open generates ambient audio (if GPU quota available)
    3. Fallback: CLAP retrieval with the optimized audio query
    """
    # Step 1: Convert scene prompt to sound-focused query
    audio_query = _make_audio_query(prompt)

    # --- Attempt 1: Stable Audio Open (free, GPU-powered) ---
    path = _stable_audio_generate(audio_query, duration=8.0)
    if path:
        return {
            "path": path, "backend": "generative",
            "model": "Stable-Audio-Open", "failed": False,
        }

    # --- Fallback: CLAP retrieval with optimized audio query ---
    logger.info("Stable Audio unavailable — using AI-enhanced CLAP retrieval")
    result = retrieve_audio(audio_query)
    result["generation_unavailable"] = True
    return result


def retrieve_image(prompt: str) -> dict:
    r = load_image_retriever().retrieve(prompt)
    return {
        "path": r.image_path, "similarity": r.similarity, "domain": r.domain,
        "failed": r.retrieval_failed, "top_5": r.top_5, "backend": "retrieval",
    }


def retrieve_audio(prompt: str) -> dict:
    r = load_audio_retriever().retrieve(prompt)
    return {
        "path": r.audio_path, "similarity": r.similarity,
        "failed": r.retrieval_failed, "top_5": r.top_5, "backend": "retrieval",
    }


def eval_coherence(text: str, image_path: str, audio_path: str) -> dict:
    return load_coherence_engine().evaluate(text=text, image_path=image_path, audio_path=audio_path)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _sc_cls(v: Optional[float]) -> str:
    if v is None: return ""
    if v >= 0.45: return "sc-high"
    if v >= 0.30: return "sc-mid"
    return "sc-low"

def _sc_badge(v: Optional[float]) -> str:
    if v is None: return ""
    if v >= 0.45: return "High"
    if v >= 0.30: return "Moderate"
    return "Low"

def score_card_html(label: str, value: Optional[float], is_class: bool = False) -> str:
    if is_class:
        badge_text = _sc_badge(value) or "N/A"
        val_display = f"{badge_text} Coherence"
        badge_html = f'<div class="sc-badge">MSCI {value:.3f}</div>' if value is not None else ""
        return (f'<div class="sc sc-class"><div class="sc-lbl">{label}</div>'
                f'<div class="sc-val">{val_display}</div>{badge_html}</div>')
    cls = _sc_cls(value)
    val_str = f"{value:.4f}" if value is not None else "\u2014"
    badge = _sc_badge(value)
    badge_html = f'<div class="sc-badge">{badge}</div>' if badge else ""
    return (f'<div class="sc {cls}"><div class="sc-lbl">{label}</div>'
            f'<div class="sc-val">{val_str}</div>{badge_html}</div>')

def sim_bar_html(name: str, val: float, mx: float = 0.6) -> str:
    pct = min(val / mx * 100, 100)
    cls = "sbf-g" if val >= 0.35 else ("sbf-y" if val >= 0.20 else "sbf-r")
    return (f'<div class="sb"><div class="sb-top"><span>{name}</span>'
            f'<span class="sb-v">{val:.4f}</span></div>'
            f'<div class="sb-track"><div class="sb-fill {cls}" style="width:{pct}%"></div></div></div>')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Multimodal Coherence AI",
        page_icon="\U0001f3a8",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar — settings first (needed for CSS choice)
    with st.sidebar:
        st.markdown("#### Configuration")

        kid_mode = st.toggle("\U0001f476 Kid Mode", value=False)

        lang = st.selectbox(
            "Language / Sprache",
            ["en", "de"],
            format_func=lambda x: {"en": "English", "de": "Deutsch"}[x],
        )

        # Select labels based on kid mode and language
        if kid_mode:
            L = UI_LABELS_KID.get(lang, UI_LABELS_KID["en"])
        else:
            L = UI_LABELS[lang]

        backend = st.selectbox(
            L["backend"],
            ["generative", "retrieval"],
            format_func=lambda x: {
                "generative": "Generative (Groq + Pollinations)",
                "retrieval": "Retrieval (CLIP + CLAP index)",
            }[x],
        )

        mode = st.selectbox(
            L["planning"],
            ["direct", "planner", "council", "extended_prompt"],
            format_func=lambda x: {
                "direct": "Direct",
                "planner": "Planner (single LLM call)",
                "council": "Council (3-way merge)",
                "extended_prompt": "Extended (3x tokens)",
            }[x],
        )

        st.divider()
        st.markdown(f"#### {L['examples']}")

        # Kid mode uses fun themed prompts; normal mode uses domain prompts
        if kid_mode:
            lang_examples = KID_EXAMPLE_PROMPTS.get(lang, KID_EXAMPLE_PROMPTS["en"])
            for dname, prompts in lang_examples.items():
                with st.expander(dname):  # already has emoji in key
                    for p in prompts:
                        if st.button(p, key=f"ex_{hash(p)}", use_container_width=True):
                            st.session_state["prompt_input"] = p
        else:
            lang_examples = EXAMPLE_PROMPTS.get(lang, EXAMPLE_PROMPTS["en"])
            domain_icons_de = {"natur": "\U0001f33f", "stadt": "\U0001f3d9\ufe0f", "wasser": "\U0001f30a", "gemischt": "\U0001f310"}
            for dname, prompts in lang_examples.items():
                icon = DOMAIN_ICONS.get(dname.lower(), domain_icons_de.get(dname.lower(), "\U0001f4cd"))
                with st.expander(f"{icon} {dname}"):
                    for p in prompts:
                        if st.button(p, key=f"ex_{hash(p)}", use_container_width=True):
                            st.session_state["prompt_input"] = p

        st.divider()
        mode_desc = {
            "direct": "Prompt used directly for all modalities",
            "planner": "LLM creates a semantic plan with image/audio prompts",
            "council": "3 LLM calls merged for richer planning",
            "extended_prompt": "Single LLM call with 3x token budget",
        }
        if backend == "generative":
            img_info = "Pollinations FLUX / Stable Horde (free)"
            aud_info = "Stable Audio / AI-matched ambience (free)"
        else:
            img_info = "CLIP retrieval (57 images)"
            aud_info = "CLAP retrieval (104 clips)"
        trans_info = "<br><b>Translation</b> opus-mt-de-en / en-de" if lang == "de" else ""
        st.markdown(
            f'<div class="sidebar-info">'
            f'<b>Text</b> HF Inference API<br>'
            f'<b>Planning</b> {mode_desc[mode]}<br>'
            f'<b>Image</b> {img_info}<br>'
            f'<b>Audio</b> {aud_info}{trans_info}<br><br>'
            f'<b>Metric</b> MSCI = 0.45 &times; s<sub>t,i</sub> + 0.45 &times; s<sub>t,a</sub><br><br>'
            f'<b>Models</b><br>'
            f'CLIP ViT-B/32 (coherence eval)<br>'
            f'CLAP HTSAT-unfused (coherence eval)'
            f'</div>', unsafe_allow_html=True)

    # Apply CSS based on mode
    if kid_mode:
        st.markdown(KID_CSS, unsafe_allow_html=True)      # kid theme (includes all needed overrides)
    else:
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)   # professional dark theme

    # Hero
    if kid_mode:
        st.markdown(
            f'<div class="kid-hero">'
            f'<div class="kid-hero-title">{L["hero_title"]}</div>'
            f'<div class="kid-hero-sub">{L["hero_sub"]}</div>'
            f'</div>', unsafe_allow_html=True)
        st.markdown(MASCOT_HTML, unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="hero-wrap">'
            f'<div class="hero-title">{L["hero_title"]}</div>'
            f'<div class="hero-sub">{L["hero_sub"]}</div>'
            f'</div>', unsafe_allow_html=True)

    # Prompt input
    default_prompt = st.session_state.get("prompt_input", "")
    prompt = st.text_area(
        "Scene", value=default_prompt, height=80,
        placeholder=L["scene_placeholder"],
        label_visibility="collapsed",
    )

    # Button + chips
    bc1, bc2 = st.columns([1, 3])
    with bc1:
        go = st.button(L["generate_btn"], type="primary", use_container_width=True, disabled=not prompt.strip())
    with bc2:
        mlbl = {"direct": "Direct", "planner": "Planner", "council": "Council", "extended_prompt": "Extended"}[mode]
        mcls = "chip-amber" if mode != "direct" else "chip-purple"
        mdot = "chip-dot-amber" if mode != "direct" else "chip-dot-purple"
        if backend == "generative":
            bchip = '<span class="chip chip-pink"><span class="chip-dot chip-dot-pink"></span>Generative</span>'
        else:
            bchip = '<span class="chip chip-purple"><span class="chip-dot chip-dot-purple"></span>Retrieval</span>'
        lang_chip = ""
        if lang == "de":
            lang_chip = '<span class="chip chip-amber"><span class="chip-dot chip-dot-amber"></span>DE \u2192 EN</span>'
        kid_chip = ""
        if kid_mode:
            kid_chip = '<span class="chip chip-green"><span class="chip-dot chip-dot-green"></span>\U0001f476 Kid</span>'
        st.markdown(
            f'<div class="chip-row">'
            f'{bchip}'
            f'<span class="chip {mcls}"><span class="chip-dot {mdot}"></span>{mlbl}</span>'
            f'<span class="chip chip-green"><span class="chip-dot chip-dot-green"></span>CLIP + CLAP</span>'
            f'{lang_chip}{kid_chip}'
            f'</div>', unsafe_allow_html=True)

    # Welcome state
    if not go and "last_result" not in st.session_state:
        if kid_mode:
            st.markdown(
                f'<div class="welcome" style="background:rgba(255,255,255,0.5);border-radius:24px;padding:3rem 2rem;">'
                f'<div class="welcome-icons">\U0001f916\u2728\U0001f3a8\u2728\U0001f3b5</div>'
                f'<div class="welcome-text" style="color:#334155;">{L["welcome_text"]}</div>'
                f'<div class="welcome-hint" style="color:#64748b;">{L["welcome_hint"]}</div>'
                f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="welcome">'
                f'<div class="welcome-icons">\U0001f3a8  \U0001f5bc\ufe0f  \U0001f50a</div>'
                f'<div class="welcome-text">{L["welcome_text"]}</div>'
                f'<div class="welcome-hint">{L["welcome_hint"]}</div>'
                f'</div>', unsafe_allow_html=True)
        return

    if go and prompt.strip():
        st.session_state["last_result"] = run_pipeline(prompt.strip(), mode, backend, lang)
        st.session_state["last_result"]["kid_mode"] = kid_mode

    if "last_result" in st.session_state:
        # Update kid_mode in case user toggled it after generation
        st.session_state["last_result"]["kid_mode"] = kid_mode
        show_results(st.session_state["last_result"])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(prompt: str, mode: str, backend: str = "generative", lang: str = "en") -> dict:
    R: dict = {"mode": mode, "backend": backend, "lang": lang, "original_prompt": prompt}
    t_all = time.time()

    # 0) Translate German → English if needed
    en_prompt = prompt
    if lang == "de":
        with st.status("\u00dcbersetze ins Englische...", expanded=True) as s:
            t0 = time.time()
            en_prompt = translate_de_to_en(prompt)
            t_trans = time.time() - t0
            R["t_translate"] = t_trans
            R["en_prompt"] = en_prompt
            s.update(label=f"Translated ({t_trans:.1f}s): {en_prompt[:80]}...", state="complete")
    else:
        R["en_prompt"] = prompt

    # 1) Text + Planning (always in English for CLIP/CLAP)
    plan_label = "Generating text..." if mode == "direct" else f"Planning ({mode}) + generating text..."
    with st.status(plan_label, expanded=True) as s:
        t0 = time.time()
        try:
            R["text"] = gen_text(en_prompt, mode)
            R["t_text"] = time.time() - t0
            has_plan = R["text"].get("plan") is not None
            lbl = f"Text ready ({R['t_text']:.1f}s)"
            if has_plan:
                lbl = f"Plan + text ready ({R['t_text']:.1f}s)"
            s.update(label=lbl, state="complete")
        except Exception as e:
            s.update(label=f"Text failed: {e}", state="error")
            R["text"] = {"text": en_prompt, "image_prompt": en_prompt, "audio_prompt": en_prompt}
            R["t_text"] = time.time() - t0

    # Translate generated text back to German for display
    if lang == "de":
        en_text = R["text"].get("text", "")
        R["text"]["text_en"] = en_text
        R["text"]["text"] = translate_en_to_de(en_text)

    ip = R["text"].get("image_prompt", en_prompt)
    ap = R["text"].get("audio_prompt", en_prompt)

    # 2) Image
    img_label = "Generating image..." if backend == "generative" else "Retrieving image..."
    with st.status(img_label, expanded=True) as s:
        t0 = time.time()
        try:
            if backend == "generative":
                R["image"] = generate_image(ip)
            else:
                R["image"] = retrieve_image(ip)
            R["t_img"] = time.time() - t0
            img_backend = R["image"].get("backend", "unknown")
            model = R["image"].get("model", "")
            if img_backend == "generative":
                lbl = f"Image generated via {model} ({R['t_img']:.1f}s)"
            else:
                sim = R["image"].get("similarity", 0)
                failed = R["image"].get("failed", False)
                lbl = f"Image retrieved (sim={sim:.3f}, {R['t_img']:.1f}s)"
                if failed:
                    lbl += " \u2014 below threshold"
            s.update(label=lbl, state="complete")
        except Exception as e:
            s.update(label=f"Image failed: {e}", state="error")
            R["image"] = None
            R["t_img"] = time.time() - t0

    # 3) Audio
    aud_label = "Generating audio..." if backend == "generative" else "Retrieving audio..."
    with st.status(aud_label, expanded=True) as s:
        t0 = time.time()
        try:
            if backend == "generative":
                R["audio"] = generate_audio(ap)
            else:
                R["audio"] = retrieve_audio(ap)
            R["t_aud"] = time.time() - t0
            aud_backend = R["audio"].get("backend", "unknown")
            model = R["audio"].get("model", "")
            if aud_backend == "generative":
                lbl = f"Audio generated via {model} ({R['t_aud']:.1f}s)"
            else:
                sim = R["audio"].get("similarity", 0)
                failed = R["audio"].get("failed", False)
                lbl = f"Audio retrieved (sim={sim:.3f}, {R['t_aud']:.1f}s)"
                if failed:
                    lbl += " \u2014 below threshold"
            s.update(label=lbl, state="complete")
        except Exception as e:
            s.update(label=f"Audio failed: {e}", state="error")
            R["audio"] = None
            R["t_aud"] = time.time() - t0

    # 4) Coherence evaluation (always use English text for CLIP/CLAP)
    with st.status("Evaluating coherence...", expanded=True) as s:
        t0 = time.time()
        try:
            imgp = R.get("image", {}).get("path") if R.get("image") else None
            audp = R.get("audio", {}).get("path") if R.get("audio") else None
            eval_text = R["text"].get("text_en", R["text"]["text"])  # English for CLIP/CLAP
            R["coherence"] = eval_coherence(eval_text, imgp, audp)
            R["t_eval"] = time.time() - t0
            msci = R["coherence"].get("scores", {}).get("msci")
            s.update(label=f"MSCI = {msci:.4f} ({R['t_eval']:.1f}s)", state="complete")
        except Exception as e:
            s.update(label=f"Eval failed: {e}", state="error")
            R["coherence"] = None
            R["t_eval"] = time.time() - t0

    R["t_total"] = time.time() - t_all
    R["prompt"] = prompt
    return R


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def show_results(R: dict):
    coh = R.get("coherence")
    sc = coh.get("scores", {}) if coh else {}
    msci = sc.get("msci")
    st_i = sc.get("st_i")
    st_a = sc.get("st_a")
    lang = R.get("lang", "en")
    kid_mode = R.get("kid_mode", False)

    if kid_mode:
        L = UI_LABELS_KID.get(lang, UI_LABELS_KID["en"])
    else:
        L = UI_LABELS.get(lang, UI_LABELS["en"])

    # Warn banner CSS class
    warn_cls = "kid-warn" if kid_mode else "warn-banner"

    # --- Score cards ---
    if kid_mode:
        st.markdown(f'<div class="kid-sec-label">{L["scores_label"]}</div>', unsafe_allow_html=True)
        # Kid verdict banner
        verdict = _kid_verdict(msci, lang)
        st.markdown(f'<div class="kid-verdict">{verdict}</div>', unsafe_allow_html=True)
        # Balloons for high coherence!
        if msci is not None and msci >= 0.40:
            st.balloons()
        cards = (
            kid_score_card("\U0001f3af Gesamt" if lang == "de" else "\U0001f3af Overall", msci, is_main=True)
            + kid_score_card("\U0001f5bc\ufe0f Text \u2192 Bild" if lang == "de" else "\U0001f5bc\ufe0f Text \u2192 Image", st_i)
            + kid_score_card("\U0001f50a Text \u2192 Ton" if lang == "de" else "\U0001f50a Text \u2192 Audio", st_a)
            + kid_score_card("\U0001f31f Sterne" if lang == "de" else "\U0001f31f Stars", msci)
        )
        st.markdown(f'<div class="kid-scores">{cards}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="sec-label">{L["scores_label"]}</div>', unsafe_allow_html=True)
        cards = (
            score_card_html("MSCI (Overall)", msci)
            + score_card_html("Text \u2192 Image", st_i)
            + score_card_html("Text \u2192 Audio", st_a)
            + score_card_html("Classification", msci, is_class=True)
        )
        st.markdown(f'<div class="scores-grid">{cards}</div>', unsafe_allow_html=True)

    # Timing strip
    tt = R.get("t_total", 0)
    sep = '<span class="t-sep">|</span>'
    trans_timing = f'{sep}<span>Translate {R.get("t_translate", 0):.1f}s</span>' if lang == "de" else ""
    timing_cls = "kid-timing" if kid_mode else "timing"
    st.markdown(
        f'<div class="{timing_cls}">'
        f'<span class="t-total">Total {tt:.1f}s</span>{sep}'
        f'{trans_timing}'
        f'<span>Text {R.get("t_text", 0):.1f}s</span>{sep}'
        f'<span>Image {R.get("t_img", 0):.1f}s</span>{sep}'
        f'<span>Audio {R.get("t_aud", 0):.1f}s</span>{sep}'
        f'<span>Eval {R.get("t_eval", 0):.1f}s</span>'
        f'</div>', unsafe_allow_html=True)

    st.markdown("---")

    # CSS class helpers for kid/normal mode
    sec_cls = "kid-sec-label" if kid_mode else "sec-label"
    text_cls = "kid-text-card" if kid_mode else "text-card"

    # Three columns: text | image | audio
    ct, ci, ca = st.columns([1.15, 1, 0.85])

    with ct:
        st.markdown(f'<div class="{sec_cls}">{L["gen_text_label"]}</div>', unsafe_allow_html=True)
        txt = R.get("text", {}).get("text", "")
        text_err = R.get("text", {}).get("text_error")
        if text_err:
            st.markdown(
                f'<div class="{warn_cls}"><b>Text gen failed</b> — using prompt as text. '
                f'({text_err})</div>',
                unsafe_allow_html=True)
        st.markdown(f'<div class="{text_cls}">{txt}</div>', unsafe_allow_html=True)
        # Show English original when in German mode
        if lang == "de":
            text_en = R.get("text", {}).get("text_en", "")
            if text_en and text_en != txt:
                with st.expander("English (original)"):
                    st.markdown(f'<div class="{text_cls}" style="opacity:0.7">{text_en}</div>',
                                unsafe_allow_html=True)

    with ci:
        st.markdown(f'<div class="{sec_cls}">{L["gen_image_label"]}</div>', unsafe_allow_html=True)
        ii = R.get("image")
        if ii and ii.get("path"):
            ip = Path(ii["path"])
            backend = ii.get("backend", "unknown")

            if backend == "retrieval" and R.get("backend") == "generative":
                sim = ii.get("similarity", 0)
                st.markdown(
                    f'<div class="{warn_cls}">Image generation unavailable '
                    f'\u2014 using CLIP retrieval (sim={sim:.3f}).</div>',
                    unsafe_allow_html=True)

            if ip.exists():
                st.image(str(ip), use_container_width=True)
                model = ii.get("model", "")
                if backend == "generative":
                    cap = f"\U0001f3a8 Pixela hat gemalt mit **{model}**" if kid_mode and lang == "de" else (
                        f"\U0001f3a8 Pixela painted with **{model}**" if kid_mode else f"Generated via **{model}**")
                    st.caption(cap)
                else:
                    sim = ii.get("similarity", 0)
                    dom = ii.get("domain", "other")
                    ic = DOMAIN_ICONS.get(dom, "\U0001f4cd")
                    st.caption(f"{ic} {dom} \u00b7 sim **{sim:.3f}** \u00b7 Retrieved")
        else:
            st.info("No image." if not kid_mode else "\U0001f3a8 Kein Bild." if lang == "de" else "\U0001f3a8 No image.")

    with ca:
        st.markdown(f'<div class="{sec_cls}">{L["gen_audio_label"]}</div>', unsafe_allow_html=True)
        ai = R.get("audio")
        if ai and ai.get("path"):
            ap = Path(ai["path"])
            backend = ai.get("backend", "unknown")

            if backend == "retrieval" and R.get("backend") == "generative":
                sim = ai.get("similarity", 0)
                if ai.get("generation_unavailable"):
                    if kid_mode:
                        msg = ("Soundo hat ein passendes Lied aus seiner Sammlung geholt!"
                               if lang == "de" else
                               "Soundo picked a matching sound from the library!")
                        st.markdown(f'<div class="{warn_cls}">{msg}</div>',
                                    unsafe_allow_html=True)
                    else:
                        sfx_err = ai.get("sfx_error", "unknown")
                        st.markdown(
                            f'<div class="{warn_cls}">ElevenLabs SFX failed: {sfx_err} '
                            f'\u2014 using CLAP retrieval (sim={sim:.3f}).</div>',
                            unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="{warn_cls}">Audio generation unavailable '
                        f'\u2014 using CLAP retrieval (sim={sim:.3f}).</div>',
                        unsafe_allow_html=True)

            if ap.exists():
                st.audio(str(ap))
                model = ai.get("model", "")
                if backend == "generative":
                    cap = f"\U0001f3b5 Soundo spielt mit **{model}**" if kid_mode and lang == "de" else (
                        f"\U0001f3b5 Soundo plays with **{model}**" if kid_mode else f"Generated via **{model}**")
                    st.caption(cap)
                else:
                    sim = ai.get("similarity", 0)
                    st.caption(f"sim **{sim:.3f}** \u00b7 Retrieved")
        else:
            st.info("No audio." if not kid_mode else "\U0001f3b5 Kein Audio." if lang == "de" else "\U0001f3b5 No audio.")

    st.markdown("---")

    # Expandable details (hidden in kid mode to keep it simple)
    if not kid_mode:
        with st.expander("Semantic Plan"):
            td = R.get("text", {})
            plan = td.get("plan")
            if plan:
                p1, p2 = st.columns(2)
                with p1:
                    dash = "\u2014"
                    dot = "\u00b7"
                    scene = plan.get("scene_summary", dash)
                    domain = plan.get("domain", dash)
                    core = plan.get("core_semantics", {})
                    setting = core.get("setting", dash)
                    tod = core.get("time_of_day", dash)
                    weather = core.get("weather", dash)
                    subjects = ", ".join(core.get("main_subjects", []))
                    st.markdown(f"**Scene** {scene}")
                    st.markdown(f"**Domain** {domain}")
                    st.markdown(f"**Setting** {setting} {dot} **Time** {tod} {dot} **Weather** {weather}")
                    st.markdown(f"**Subjects** {subjects}")
                with p2:
                    st.markdown("**Image prompt**")
                    st.code(td.get("image_prompt", ""), language=None)
                    st.markdown("**Audio prompt**")
                    st.code(td.get("audio_prompt", ""), language=None)
            else:
                mode = R.get("mode", "direct")
                if mode == "direct":
                    st.write("Direct mode \u2014 no semantic plan. Prompt used as-is for all modalities.")
                else:
                    st.write(f"Planning ({mode}) did not produce a valid plan. Fell back to direct mode.")

        with st.expander("Generation Details"):
            r1, r2 = st.columns(2)
            with r1:
                ii = R.get("image")
                if ii:
                    backend = ii.get("backend", "unknown")
                    model = ii.get("model", "")
                    if backend == "generative":
                        st.markdown(f"**Image** generated via **{model}**")
                        st.markdown(f"Prompt: *{R.get('text', {}).get('image_prompt', '')}*")
                    elif ii.get("top_5"):
                        st.markdown("**Image** (retrieval fallback)")
                        bars = "".join(sim_bar_html(n, s) for n, s in ii["top_5"])
                        st.markdown(bars, unsafe_allow_html=True)
                else:
                    st.write("No image data.")
            with r2:
                ai = R.get("audio")
                if ai:
                    backend = ai.get("backend", "unknown")
                    model = ai.get("model", "")
                    if backend == "generative":
                        st.markdown(f"**Audio** generated via **{model}**")
                        st.markdown(f"Prompt: *{R.get('text', {}).get('audio_prompt', '')}*")
                    elif ai.get("top_5"):
                        st.markdown("**Audio** (retrieval fallback)")
                        bars = "".join(sim_bar_html(n, s) for n, s in ai["top_5"])
                        st.markdown(bars, unsafe_allow_html=True)
                else:
                    st.write("No audio data.")

        with st.expander("Full Coherence Report"):
            if coh:
                st.json(coh)
            else:
                st.write("No data.")
    else:
        # Kid mode: simple "how it works" expander instead of technical details
        label_how = "\U0001f914 Wie funktioniert das?" if lang == "de" else "\U0001f914 How does it work?"
        with st.expander(label_how):
            if lang == "de":
                st.markdown(
                    "1. **Textino** \U0001f916 liest deine Beschreibung und schreibt eine Geschichte\n"
                    "2. **Pixela** \U0001f3a8 malt ein Bild, das zur Geschichte passt\n"
                    "3. **Soundo** \U0001f3b5 erzeugt Ger\u00e4usche und Musik dazu\n"
                    "4. Dann pr\u00fcfen wir, ob alles gut zusammenpasst! \u2b50"
                )
            else:
                st.markdown(
                    "1. **Textino** \U0001f916 reads your description and writes a story\n"
                    "2. **Pixela** \U0001f3a8 paints a picture that matches the story\n"
                    "3. **Soundo** \U0001f3b5 creates sounds and music for it\n"
                    "4. Then we check if everything fits together! \u2b50"
                )


if __name__ == "__main__":
    main()

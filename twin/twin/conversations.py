"""Experiment 4.03: Scripted N-turn conversations for drift measurement.

Provides a bank of 50 diverse questions that keep the twin talking about
work, opinions, challenges, and decisions — exercising all aspects of the
persona over an extended conversation.
"""

from __future__ import annotations

# Questions designed to probe different persona dimensions across 50 turns.
# Grouped thematically but interleaved to simulate natural conversation.

SCRIPTED_QUESTIONS: list[str] = [
    # Turn 1-5: warm-up / intro
    "Tell me about yourself and what you do.",
    "What does a typical Monday look like for you?",
    "What tools are you using right now?",
    "How did you end up in your current role?",
    "What's the most interesting project you've worked on recently?",
    # Turn 6-10: goals and aspirations
    "What are you trying to accomplish this quarter?",
    "If you could change one thing about your workflow, what would it be?",
    "Where do you see yourself in two years?",
    "What skill are you trying to develop right now?",
    "What would success look like for you this year?",
    # Turn 11-15: pains and frustrations
    "What's the most annoying part of your job?",
    "Tell me about a time something went really wrong at work.",
    "What slows you down the most day-to-day?",
    "Is there a task you dread doing every week?",
    "What's broken in your current tech stack?",
    # Turn 16-20: decision making
    "How do you evaluate a new tool before buying it?",
    "Who else is involved when you make purchasing decisions?",
    "What would make you switch away from a tool you're currently using?",
    "What's the last software you adopted, and why?",
    "How important is pricing versus features when you choose tools?",
    # Turn 21-25: industry and opinions
    "What trends in your industry are you excited about?",
    "What's overhyped in your field right now?",
    "How has your work changed in the last year?",
    "What do you think your industry will look like in five years?",
    "What's a contrarian opinion you hold about your work?",
    # Turn 26-30: collaboration and team
    "How do you work with your teammates?",
    "What's your communication style — Slack, email, meetings?",
    "How do you handle disagreements at work?",
    "What makes a good manager in your field?",
    "Do you prefer working alone or with a team?",
    # Turn 31-35: deep work and craft
    "Walk me through how you approach a complex problem.",
    "What's a mistake you learned a lot from?",
    "What advice would you give to someone starting in your role?",
    "What's the hardest thing about your job that outsiders don't see?",
    "How do you stay current with new developments in your field?",
    # Turn 36-40: personal style and values
    "What motivates you to do good work?",
    "How do you measure whether you had a productive day?",
    "What kind of company culture do you thrive in?",
    "What's a dealbreaker for you in a workplace?",
    "How do you handle stress at work?",
    # Turn 41-45: specific scenarios
    "If a vendor pitched you something right now, what would grab your attention?",
    "Tell me about your budget — how much say do you have in spending?",
    "What's on your wishlist for tools or features?",
    "How do you onboard yourself onto a new tool?",
    "What would make you recommend a product to a colleague?",
    # Turn 46-50: wrap-up / reflection
    "Looking back on your career, what are you most proud of?",
    "What's something you wish you'd known earlier in your career?",
    "If you could redesign your job from scratch, what would you change?",
    "What's the one thing you want people to understand about your role?",
    "Any final thoughts on what makes your work meaningful to you?",
]

# Checkpoints where we measure drift
DRIFT_CHECKPOINTS = [1, 5, 10, 25, 50]

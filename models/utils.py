"""
Utility functions for text processing and statistics
"""


def calculate_compression_ratio(original_text, summary_text):
    """Calculate compression ratio"""
    if len(original_text) == 0:
        return 0.0
    return len(summary_text) / len(original_text)


def get_statistics(original_text, summary_text):
    """Calculate statistics for a summary"""
    stats = {
        'original_length': len(original_text),
        'summary_length': len(summary_text),
        'compression_ratio': calculate_compression_ratio(original_text, summary_text),
        'reduction_percentage': (1 - calculate_compression_ratio(original_text, summary_text)) * 100
    }
    return stats


def get_example_notes():
    """Return example clinical notes for demo"""
    examples = {
        "Chest Pain": """17 year old male presents to clinic complaining of chest pain for 2 days. Patient states pain is sharp and worse with deep breathing. Denies fever, cough, or shortness of breath. Physical examination reveals clear lungs bilaterally, normal heart sounds, vitals stable. Blood pressure 118/72, heart rate 78, temperature 98.6. Assessment indicates likely musculoskeletal chest pain, possibly costochondritis. Plan includes ibuprofen 400mg every 6 hours as needed for pain, return if symptoms worsen.""",

        "Fever": """45 year old female presents with fever and chills for 3 days. Temperature reported as 101.5F at home. Associated symptoms include body aches, fatigue, and mild headache. Denies cough, sore throat, or shortness of breath. Physical exam shows temperature 100.8F, other vitals within normal limits. Assessment: likely viral syndrome. Plan: symptomatic treatment with acetaminophen, rest, hydration.""",

        "Headache": """28 year old presents with severe headache for 24 hours. Describes pain as throbbing, bilateral, worse with movement. No visual changes or neck stiffness. Vital signs normal. Neurological exam unremarkable. Assessment: tension headache. Plan: rest, hydration, NSAIDs as needed, follow up if persistent."""
    }
    return examples
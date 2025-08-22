import librosa
from ..models.ultravox_model import get_pipe, is_model_loaded

def create_evaluation_prompt():
    """Create the evaluation prompt for interview analysis"""
    return '''You are an advanced audio analysis model tasked with evaluating a candidate’s interview introduction audio file for a professional role. Analyze the audio based on the following criteria and provide a decision to shortlist or reject the candidate, along with a probability value and detailed remarks explaining the evaluation.

Input: Audio file containing the candidate’s interview introduction.

Criteria for Analysis:

Emotions:
Confidence Level & Nervousness: Assess the candidate’s confidence by detecting the frequency of filler words (e.g., “um,” “uh,” “you know”). Frequent use (more than 3 instances per 30 seconds) indicates nervousness, while minimal use suggests confidence.
Pitch and Tone: Evaluate the pitch and tone of the candidate’s voice. A bold, steady, and consistent tone reflects confidence, while a shaky, high-pitched, or inconsistent tone may indicate nervousness.
Communication:
Clarity of Speech: Check for accurate pronunciation of words, especially technical terms (e.g., “data analysis” should not be pronounced as “datanalysis”). Mispronunciations indicate poor clarity.
Fluency: Measure pauses in speech. A fluent delivery has no more than 1 pause (lasting less than 2 seconds) per 10 seconds or per sentence. Excessive pauses or hesitations indicate lower fluency.
Smooth Transitions: Assess the flow between sentences. Transitions should be seamless, with no abrupt pauses or awkward shifts in topic or tone.
Accent: Evaluate the accent for neutrality and intelligibility. The accent should not hinder comprehension for a general professional audience.
Grammatical Accuracy: Analyze the grammatical correctness of spoken sentences. For example, “The team was successful” is correct, while “The team were successful” is incorrect. Flag any grammatical errors.
Output Requirements:

Decision: State whether the candidate is “Shortlisted” or “Rejected” based on the overall evaluation.
Probability Value: Provide a confidence score (0–100%) indicating the likelihood of the decision (e.g., 85% for Shortlist or 30% for Reject).
Remarks: Provide a concise explanation (100–150 words) detailing the analysis for each criterion, highlighting strengths and weaknesses. Specify any detected issues (e.g., frequent filler words, mispronunciations, grammatical errors) and their impact on the decision.
Instructions:

Analyze the audio holistically, weighing all criteria equally unless a critical failure (e.g., unintelligible accent or frequent mispronunciations) significantly impacts suitability.
Use objective language in remarks, focusing on measurable observations (e.g., “Detected 5 filler words in 30 seconds”).
Ensure the decision aligns with the analysis, prioritizing candidates who demonstrate confidence, clear communication, and grammatical accuracy.
Example Output Format:

Decision: Shortlisted
Probability: 85%
Remarks: The candidate exhibited strong confidence with minimal filler words (1 “um” in 30 seconds) and a bold, steady tone. Speech clarity was excellent, with accurate pronunciation of technical terms like “data analysis.” Fluency was high, with no pauses exceeding 1 second per sentence. Transitions between sentences were smooth, maintaining a coherent flow. The accent was neutral and easily understandable. However, one minor grammatical error was noted (“the team were successful”). Overall, the candidate’s performance was strong, justifying a shortlist decision.
'''

def process_audio_file(audio_path, max_new_tokens=10000):
    """
    Process an audio file and return the evaluation results.
    
    Args:
        audio_path (str): Path to the audio file
        max_new_tokens (int): Maximum number of tokens to generate
        
    Returns:
        dict: The model output containing the evaluation
    """
    try:
        # Load audio file
        print(f"Loading audio file: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=16000)
        print(f"Audio loaded successfully. Duration: {len(audio)/sr:.2f} seconds")
        
        # Check if model is already loaded
        if not is_model_loaded():
            print("🔄 Loading Ultravox model into VRAM...")
        else:
            print("✅ Using pre-loaded Ultravox model from VRAM")
        
        # Get the model pipeline (singleton - loads only once)
        pipe = get_pipe()
        
        # Create conversation turns
        turns = [
            {
                "role": "system",
                "content": create_evaluation_prompt()
            },
        ]
        
        # Run inference
        print("Running inference...")
        output = pipe({'audio': audio, 'turns': turns, 'sampling_rate': sr}, max_new_tokens=max_new_tokens)
        
        return output
        
    except Exception as e:
        print(f"Error processing audio file {audio_path}: {str(e)}")
        return None

def main():
    """Main function to demonstrate usage"""
    # Example usage - you can change this audio path without reloading the model
    audio_path = "Audios/ATSID00897933_introduction.wav"
    
    # Process the audio file
    result = process_audio_file(audio_path)
    
    if result:
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(result)
    else:
        print("Failed to process audio file")

if __name__ == "__main__":
    main() 
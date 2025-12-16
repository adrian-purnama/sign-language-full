/**
 * Google Gemini API integration for emergency mode
 * Processes sign language images directly in the frontend
 */

export interface GeminiResponse {
  translation: string;
  confidence?: number;
}

// API key should be entered by user in Settings panel

/**
 * Call Google Gemini API with image data
 */
export const translateWithGemini = async (
  imageData: string, // base64 image data
  apiKey: string
): Promise<GeminiResponse> => {
    apiKey = "AIzaSyCQsaOygq9UMTFttPSQrIHQMTJd9LJ6sXs";
  if (!apiKey) {
    throw new Error('Gemini API key is required');
  }

  // Remove data URL prefix if present
  let base64Image = imageData;
  if (imageData.includes(',')) {
    base64Image = imageData.split(',')[1];
  }

  try {
    // Use gemini-2.5-flash-lite model (higher rate limits: 10 RPM vs 5 RPM)
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash-lite:generateContent?key=${apiKey}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          contents: [
            {
              parts: [
                {
                  text: 'You are a sign language recognition expert. Analyze this image carefully and identify what sign language gesture the person is making. Return ONLY the English word that this sign represents. Return a single word in lowercase. If you cannot clearly identify a sign or the person is not making a sign, return exactly three dots: "...". Do not include any explanation, punctuation, or additional text - only the word or "...".'
                },
                {
                  inline_data: {
                    mime_type: 'image/jpeg',
                    data: base64Image
                  }
                }
              ]
            }
          ],
          generationConfig: {
            temperature: 0.1,
            topK: 1,
            topP: 1,
            maxOutputTokens: 10,
          }
        })
      }
    );

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      
      // Handle rate limiting specifically
      if (response.status === 429) {
        throw new Error('Rate limit exceeded. Please wait a moment and try again.');
      }
      
      throw new Error(errorData.error?.message || `API error: ${response.status}`);
    }

    const data = await response.json();
    
    console.log('Gemini API response:', data); // Debug log
    
    if (!data.candidates || !Array.isArray(data.candidates) || data.candidates.length === 0) {
      throw new Error('Invalid response from Gemini API: no candidates');
    }

    const candidate = data.candidates[0];
    if (!candidate || !candidate.content) {
      throw new Error('Invalid response from Gemini API: no content in candidate');
    }

    if (!candidate.content.parts || !Array.isArray(candidate.content.parts) || candidate.content.parts.length === 0) {
      throw new Error('Invalid response from Gemini API: no parts in content');
    }

    const text = candidate.content.parts[0]?.text?.trim() || '...';
    
    // Extract word - clean up the response
    let word = text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ') // Remove punctuation
      .trim()
      .split(/\s+/)[0]; // Get first word
    
    // If response is just dots or empty, return ...
    if (!word || word === '' || word.includes('...') || word === 'dot') {
      word = '...';
    }
    
    console.log('Gemini raw response:', text, '-> cleaned:', word);
    
    return {
      translation: word,
      confidence: word === '...' ? 0.0 : 0.85 // Higher confidence when word is detected
    };
  } catch (error) {
    console.error('Gemini API error:', error);
    throw error;
  }
};


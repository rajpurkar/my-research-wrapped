import { useState, useEffect } from 'react';
import '../styles/YearInReview.css';

// Import the narrative data using Vite's glob import
const narrativeFiles = import.meta.glob<NarrativeData>('../outputs/year_in_review_narrative.json', { 
  eager: true,
  import: 'default'
});

interface TopicSection {
  name: string;
  content: string;
}

interface NarrativeData {
  introduction: string;
  topics: TopicSection[];
}

export function YearInReview() {
  const [narrative, setNarrative] = useState<NarrativeData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadNarrative = async () => {
      try {
        const filePath = Object.keys(narrativeFiles)[0];
        if (!filePath) {
          throw new Error('Narrative file not found');
        }

        const data = narrativeFiles[filePath];
        if (!data || typeof data.introduction !== 'string' || !Array.isArray(data.topics)) {
          throw new Error('Invalid narrative data format');
        }

        setNarrative(data);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to load narrative';
        setError(errorMessage);
        console.error('Error loading narrative:', err);
      }
    };

    loadNarrative();
  }, []);

  if (error) {
    return (
      <div className="error-container">
        <div className="error-content">
          <h2>Error Loading Research Narrative</h2>
          <p>{error}</p>
          <p>Please ensure the narrative file has been generated and is accessible.</p>
        </div>
      </div>
    );
  }

  if (!narrative) {
    return (
      <div className="loading-container">
        <div className="loading-content">Loading research narrative...</div>
      </div>
    );
  }

  return (
    <div className="year-in-review">
      <header className="review-header">
        <h1>My Research Wrapped 🎯✨</h1>
        <p className="subtitle">Your academic journey, beautifully visualized</p>
      </header>
      
      <div className="year-review-container">
        <div className="year-review-content">
          {/* Introduction */}
          <section className="introduction-section">
            <h1>Research Year in Review</h1>
            {narrative.introduction.split('\n').map((paragraph, index) => (
              paragraph.trim() && (
                <p key={`intro-${index}`} className="intro-paragraph">
                  {paragraph}
                </p>
              )
            ))}
          </section>

          {/* Topics */}
          <section className="topics-section">
            {narrative.topics.map((topic, topicIndex) => (
              <article key={`topic-${topicIndex}`} className="topic-article">
                <h2 className="topic-title">{topic.name}</h2>
                <div className="topic-content">
                  {topic.content.split('\n').map((paragraph, paraIndex) => (
                    paragraph.trim() && (
                      <p key={`topic-${topicIndex}-para-${paraIndex}`} className="topic-paragraph">
                        {paragraph}
                      </p>
                    )
                  ))}
                </div>
              </article>
            ))}
          </section>
        </div>
      </div>
    </div>
  );
} 
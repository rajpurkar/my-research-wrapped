import { useState, useEffect, useRef } from 'react'
import React from 'react'
import { Helmet } from 'react-helmet-async'

interface Author {
  full_name: string
  normalized_name: string
}

interface Paper {
  title: string
  summary: string
  weight: number
  role: string
  file_path: string
  authors: Author[]
}

interface Topic {
  name: string
  synthesis: string
  papers: Paper[]
}

interface NarrativeData {
  author: string
  introduction: string
  topics: Topic[]
}

const normalizeName = (name: string) => {
  const specialCases: { [key: string]: string } = {
    'Hongyu Zhou': 'Hong Yu Zhou',
    'Rajpurkar Pranav': 'Pranav Rajpurkar',
    // Add more special cases if needed
  };

  if (specialCases[name]) return specialCases[name];

  const parts = name.trim().split(/\s+/);

  if (parts.length === 2) {
    const [firstName, lastName] = parts;

    // Check if the name is in "Last First" format and flip it
    // You can add more robust checks here if necessary
    return `${firstName} ${lastName}`;
  } else {
    // Leave names with more than two parts or one part as is
    return name;
  }
};

// Add this new component above the App function
const TopicNode: React.FC<{
  index: number;
  name: string;
  onClick: () => void;
}> = ({ index, name, onClick }) => {
  return (
    <div className="topic-node-wrapper">
      <div className="topic-node-reflection"></div>
      <button className="topic-node" onClick={onClick}>
        <div className="topic-node-content">
          <span className="topic-number">0{index + 1}</span>
          <span className="topic-name">{name}</span>
        </div>
        <div className="node-glow"></div>
      </button>
    </div>
  );
};

function App() {
  const [narrative, setNarrative] = useState<NarrativeData | null>(null)
  const topicsRef = useRef<HTMLDivElement>(null)
  const [selectedPaper, setSelectedPaper] = useState<{topicIndex: number, paperIndex: number} | null>(null)
  const coAuthorFreq = useRef(new Map<string, number>())
  const nameVariants = useRef(new Map<string, string>())
  const [mainAuthorNormalized, setMainAuthorNormalized] = useState<string>('')

  // Function to get normalized name considering variants
  const getNormalizedName = (name: string) => {
    const normalizedName = normalizeName(name);
    return nameVariants.current.get(normalizedName) || normalizedName;
  };

  useEffect(() => {
    // Load narrative data
    const loadNarrative = async () => {
      try {
        const narrativeData = await import('./outputs/narrative.json')
        setNarrative(narrativeData.default)

        // Normalize the main author's name
        const normalizedMainAuthor = normalizeName(narrativeData.default.author);
        setMainAuthorNormalized(normalizedMainAuthor);

        // Temporary maps to accumulate counts and name variants
        const tempCoAuthorFreq = new Map<string, number>();
        const tempNameVariants = new Map<string, string>();

        // Calculate co-author frequencies
        narrativeData.default.topics.forEach((topic: Topic) => {
          topic.papers.forEach((paper: Paper) => {
            paper.authors.forEach((author: Author) => {
              const normalizedName = normalizeName(author.normalized_name);
              if (normalizedName !== normalizedMainAuthor) {
                // Update frequency for the normalized name
                const count = tempCoAuthorFreq.get(normalizedName) || 0;
                tempCoAuthorFreq.set(normalizedName, count + 1);

                // Map all name variants to the normalized name
                tempNameVariants.set(author.normalized_name, normalizedName);
              }
            });
          });
        });

        // Update coAuthorFreq and nameVariants
        coAuthorFreq.current = new Map(
          Array.from(tempCoAuthorFreq.entries()).filter(([_, count]) => count >= 2)
        );
        nameVariants.current = tempNameVariants;
      } catch (error) {
        console.error('Error loading narrative data:', error)
      }
    }

    loadNarrative()
  }, [])

  const scrollToTopic = (index: number) => {
    const element = document.getElementById(`topic-${index}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  // Generate more stars
  const generateStars = () => {
    const stars = [];
    for (let i = 0; i < 12; i++) {
      const delay = Math.random() * 4;
      const top = Math.random() * 100;
      const left = Math.random() * 100;
      stars.push(
        <div
          key={i}
          className="star"
          style={{
            top: `${top}%`,
            left: `${left}%`,
            animationDelay: `${delay}s`
          }}
        />
      );
    }
    return stars;
  };

  // Generate sparkles
  const generateSparkles = () => {
    const sparkles = [];
    for (let i = 0; i < 20; i++) {
      const delay = Math.random() * 4;
      const top = Math.random() * 100;
      const left = Math.random() * 100;
      const tx = (Math.random() - 0.5) * 100; // Random x translation
      const ty = (Math.random() - 0.5) * 100; // Random y translation
      sparkles.push(
        <div
          key={`sparkle-${i}`}
          className="sparkle"
          style={{
            top: `${top}%`,
            left: `${left}%`,
            animationDelay: `${delay}s`,
            '--tx': `${tx}px`,
            '--ty': `${ty}px`
          } as React.CSSProperties}
        />
      );
    }
    return sparkles;
  };

  if (!narrative) {
    return (
      <div className="loading-screen">
        <div className="loading-content">
          <div className="loading-spinner"></div>
          <p>Analyzing research impact...</p>
        </div>
      </div>
    )
  }

  return (
    <div id="root">
      <Helmet>
        <title>Research Year Wrapped 2024 | {narrative?.author || 'Loading...'}</title>
        <meta name="description" content="An interactive visualization of research contributions and impact throughout the year." />
        <meta property="og:title" content={`Research Year Wrapped 2024 | ${narrative?.author || 'Loading...'}`} />
        <meta property="og:description" content="An interactive visualization of research contributions and impact throughout the year." />
        <meta property="og:type" content="website" />
      </Helmet>

      <div className="app-container">
        <header className="app-header">
          <div className="header-content">
            <h1>MyResearchWrapped2024 🎁</h1>
            <div className="header-meta">
              <span className="year-tag">2024</span>
              <iframe
                src="https://ghbtns.com/github-btn.html?user=rajpurkar&repo=my-research-wrapped&type=star&count=true"
                frameBorder="0"
                scrolling="0"
                width="150"
                height="20"
                title="GitHub"
              />
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <section className="hero-section">
          <div className="stars-container">
            {generateStars()}
            {generateSparkles()}
          </div>
          <div className="hero-content">
            <h2 className="author-name">{narrative.author}</h2>
            <h1>My Research Wrapped 2024</h1>
            <div className="author-stats">
              {(() => {
                const totalPapers = narrative.topics.reduce((sum, topic) => sum + topic.papers.length, 0);
                const uniqueCoAuthors = new Set();
                narrative.topics.forEach(topic => {
                  topic.papers.forEach(paper => {
                    paper.authors.forEach(author => {
                      if (author.normalized_name !== narrative.author) {
                        uniqueCoAuthors.add(author.normalized_name);
                      }
                    });
                  });
                });

                return (
                  <>
                    <span className="stat-item">
                      <span className="stat-value">{totalPapers}</span>
                      <span className="stat-label">papers</span>
                    </span>
                    <span className="stat-divider">·</span>
                    <span className="stat-item">
                      <span className="stat-value">{uniqueCoAuthors.size}</span>
                      <span className="stat-label">collaborators</span>
                    </span>
                  </>
                );
              })()}
            </div>
            
            {/* Topics Grid */}
            <div className="topics-highlight">
              <div className="topics-grid-background">
                <div className="grid-lines horizontal"></div>
                <div className="grid-lines vertical"></div>
              </div>
              <div className="topics-nodes-container">
                {narrative.topics.map((topic, index) => (
                  <TopicNode
                    key={index}
                    index={index}
                    name={topic.name}
                    onClick={() => scrollToTopic(index)}
                  />
                ))}
                <div className="connecting-lines">
                  {/* Add subtle connecting lines between nodes */}
                  {narrative.topics.map((_, index) => (
                    index < narrative.topics.length - 1 && (
                      <div key={`line-${index}`} className="connection-line"></div>
                    )
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Narrative Section */}
        <section className="narrative-text">
          <div className="narrative-text-inner">
            {narrative.introduction.split('\n').map((paragraph, index) => (
              paragraph.trim() && (
                <p key={`intro-${index}`} className="narrative-paragraph">
                  {paragraph}
                </p>
              )
            ))}
          </div>
        </section>

        {/* Main Content */}
        <main className="main-content">
          {/* Co-authors Section */}
          {(() => {
            // Get all authors with at least 2 papers
            const significantCoAuthors = Array.from(coAuthorFreq.current.entries()).sort(
              (a, b) => b[1] - a[1]
            );

            // Group authors by count
            const countGroups: [string, number][][] = [];
            let currentCount = -1;
            let currentGroup: [string, number][] = [];

            significantCoAuthors.forEach(coAuthor => {
              if (coAuthor[1] !== currentCount) {
                if (currentGroup.length > 0) {
                  countGroups.push(currentGroup);
                }
                currentGroup = [coAuthor];
                currentCount = coAuthor[1];
              } else {
                currentGroup.push(coAuthor);
              }
            });
            if (currentGroup.length > 0) {
              countGroups.push(currentGroup);
            }

            // Split groups into three tiers
            const tier1: [string, number][] = [];
            const tier2: [string, number][] = [];
            const tier3: [string, number][] = [];
            
            const totalAuthors = significantCoAuthors.length;
            let currentSum = 0;
            
            countGroups.forEach(group => {
              if (currentSum < totalAuthors / 3) {
                tier1.push(...group);
              } else if (currentSum < (totalAuthors * 2) / 3) {
                tier2.push(...group);
              } else {
                tier3.push(...group);
              }
              currentSum += group.length;
            });

            return (
              <section className="coauthors-section">
                <div className="coauthors-container">
                  <div className="coauthors-grid-lines"></div>
                  <h2 className="coauthors-title">FREQUENT CO-AUTHORS</h2>
                  <div className="coauthors-subtitle">
                    {significantCoAuthors.length} co-authors with 2+ collaborations
                  </div>

                  {tier1.length > 0 && (
                    <div className="coauthors-tier tier-1">
                      {tier1.map(([name, count], idx) => (
                        <React.Fragment key={name}>
                          {idx > 0 && <span className="dot">•</span>}
                          <span className="coauthor">
                            {name}
                            <span className="paper-count">{count}</span>
                          </span>
                        </React.Fragment>
                      ))}
                    </div>
                  )}

                  {tier2.length > 0 && (
                    <div className="coauthors-tier tier-2">
                      {tier2.map(([name, count], idx) => (
                        <React.Fragment key={name}>
                          {idx > 0 && <span className="dot">•</span>}
                          <span className="coauthor">
                            {name}
                            <span className="paper-count">{count}</span>
                          </span>
                        </React.Fragment>
                      ))}
                    </div>
                  )}

                  {tier3.length > 0 && (
                    <div className="coauthors-tier tier-3">
                      {tier3.map(([name, count], idx) => (
                        <React.Fragment key={name}>
                          {idx > 0 && <span className="dot">•</span>}
                          <span className="coauthor">
                            {name}
                            <span className="paper-count">{count}</span>
                          </span>
                        </React.Fragment>
                      ))}
                    </div>
                  )}
                </div>
              </section>
            );
          })()}

          {/* Research Areas Grid */}
          <section className="research-areas">
            <div className="section-header">
              <h2>Research Areas</h2>
              <p className="section-subtitle">Key themes and contributions from the year</p>
            </div>
            
            <div className="topics-grid" ref={topicsRef}>
              {narrative.topics.map((topic, index) => (
                <div id={`topic-${index}`} key={index} className="topic-group">
                  <div className={`topic-card topic-variant-${index % 4}`}>
                    <div className="topic-content">
                      <div className="topic-header">
                        <span className="topic-number">0{index + 1}</span>
                        <h3>{topic.name}</h3>
                      </div>
                      <p className="topic-synthesis">{topic.synthesis}</p>
                    </div>
                  </div>

                  <div className="papers-grid">
                    {topic.papers.map((paper, pIndex) => (
                      <div
                        key={`${index}-paper-${pIndex}`}
                        className={`paper-item ${selectedPaper?.topicIndex === index && selectedPaper?.paperIndex === pIndex ? 'active' : ''}`}
                        onClick={() => setSelectedPaper({ topicIndex: index, paperIndex: pIndex })}
                      >
                        <div className="paper-item-content">
                          <h4 className="paper-title">{paper.title}</h4>
                          <p className="paper-summary">{paper.summary}</p>
                          <div className="paper-authors">
                            {paper.authors.map((author, aIndex) => (
                              <span key={aIndex} className="author-tag">
                                {getNormalizedName(author.normalized_name)}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>
        </main>
      </div>
    </div>
  )
}

export default App

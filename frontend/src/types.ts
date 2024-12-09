export interface Paper {
  id: string;
  title: string;
  authors: string[];
  year: number;
  topics: string[];
  citations: number;
}

export interface Topic {
  id: string;
  name: string;
  papers: string[];
} 
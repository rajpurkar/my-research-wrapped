import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { Paper, Topic } from '@/types';

interface ResearchVizProps {
  papers: Paper[];
  topics: Topic[];
  selectedTopic: string | null;
}

interface Node extends d3.SimulationNodeDatum {
  id: string;
  title: string;
  citations: number;
  topics: string[];
  radius: number;
  x?: number;
  y?: number;
}

interface Link extends d3.SimulationLinkDatum<Node> {
  value: number;
  source: Node;
  target: Node;
}

// Drag behavior
function drag(simulation: d3.Simulation<Node, undefined>) {
  function dragstarted(event: d3.D3DragEvent<SVGCircleElement, Node, Node>) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }

  function dragged(event: d3.D3DragEvent<SVGCircleElement, Node, Node>) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }

  function dragended(event: d3.D3DragEvent<SVGCircleElement, Node, Node>) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }

  return d3.drag<SVGCircleElement, Node>()
    .on('start', dragstarted)
    .on('drag', dragged)
    .on('end', dragended);
}

export default function ResearchViz({ papers, topics, selectedTopic }: ResearchVizProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || papers.length === 0) return;

    // Clear previous visualization
    d3.select(svgRef.current).selectAll('*').remove();

    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;

    // Create SVG
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    // Create nodes from papers
    const nodes: Node[] = papers.map(paper => ({
      id: paper.id,
      title: paper.title,
      citations: paper.citations,
      topics: paper.topics,
      radius: Math.sqrt(paper.citations + 10) * 3, // Scale node size by citations
    }));

    // Create links between papers that share topics
    const links: Link[] = [];
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const sharedTopics = nodes[i].topics.filter(t => 
          nodes[j].topics.includes(t)
        );
        if (sharedTopics.length > 0) {
          links.push({
            source: nodes[i],
            target: nodes[j],
            value: sharedTopics.length,
          });
        }
      }
    }

    // Create color scale for topics
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10)
      .domain(topics.map(t => t.id));

    // Create force simulation
    const simulation = d3.forceSimulation<Node>(nodes)
      .force('link', d3.forceLink<Node, Link>(links).id(d => d.id))
      .force('charge', d3.forceManyBody().strength(-100))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide<Node>().radius(d => d.radius + 2));

    // Create links
    const link = svg.append('g')
      .selectAll<SVGLineElement, Link>('line')
      .data(links)
      .join('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', d => Math.sqrt(d.value));

    // Create nodes
    const node = svg.append('g')
      .selectAll<SVGCircleElement, Node>('circle')
      .data(nodes)
      .join('circle')
      .attr('r', d => d.radius)
      .attr('fill', d => {
        const mainTopic = selectedTopic && d.topics.includes(selectedTopic)
          ? selectedTopic
          : d.topics[0];
        return colorScale(mainTopic);
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .call(drag(simulation));

    // Add tooltips
    node.append('title')
      .text(d => `${d.title}\nCitations: ${d.citations}\nTopics: ${d.topics.join(', ')}`);

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x!)
        .attr('y1', d => d.source.y!)
        .attr('x2', d => d.target.x!)
        .attr('y2', d => d.target.y!);

      node
        .attr('cx', d => d.x!)
        .attr('cy', d => d.y!);
    });

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [papers, topics, selectedTopic]);

  return (
    <svg
      ref={svgRef}
      className="research-viz"
      style={{ width: '100%', height: '100%' }}
    />
  );
} 
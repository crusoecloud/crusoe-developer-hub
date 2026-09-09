export interface HostConfig {
  name: string;
  color: string;
  personality: string;
}

export interface TranscriptLine {
  host: 'a' | 'b';
  name: string;
  color: string;
  text: string;
  round: number;
  total_rounds: number;
}

export interface Episode {
  id: string;
  topic: string;
  angle?: string;
  transcript: TranscriptLine[];
  createdAt: string;
  hostA: HostConfig;
  hostB: HostConfig;
}

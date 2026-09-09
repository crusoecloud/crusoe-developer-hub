export interface PersonaMini {
  name: string;
  color: string;
  emoji: string;
}

export interface PersonaPair {
  id: string;
  label: string;
  a: PersonaMini;
  b: PersonaMini;
}

export type Speaker = 'a' | 'b';

export interface StartEvent {
  event: 'start';
  topic: string;
  total_rounds: number;
  persona_a: PersonaMini;
  persona_b: PersonaMini;
}

export interface ChunkEvent {
  speaker: Speaker;
  name: string;
  color: string;
  chunk: string;
  round: number;
  total_rounds: number;
  done: false;
}

export interface DoneEvent {
  speaker: Speaker;
  name: string;
  color: string;
  text: string;
  round: number;
  total_rounds: number;
  done: true;
}

export interface ErrorEvent {
  error: string;
}

export type DebateEvent = StartEvent | ChunkEvent | DoneEvent | ErrorEvent;

export interface Turn {
  speaker: Speaker;
  name: string;
  color: string;
  text: string;
  round: number;
  done: boolean;
}

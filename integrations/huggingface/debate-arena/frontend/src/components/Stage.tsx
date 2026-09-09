import type { PersonaMini, Speaker } from '../types';
import Bubble from './Bubble';
import Character from './Character';

interface Props {
  personaA: PersonaMini | null;
  personaB: PersonaMini | null;
  activeSpeaker: Speaker | null;
  isStreaming: boolean;
  currentTextA: string;
  currentTextB: string;
  bubbleVisibleA: boolean;
  bubbleVisibleB: boolean;
  turnKeyA: number;
  turnKeyB: number;
}

export default function Stage({
  personaA,
  personaB,
  activeSpeaker,
  isStreaming,
  currentTextA,
  currentTextB,
  bubbleVisibleA,
  bubbleVisibleB,
  turnKeyA,
  turnKeyB,
}: Props) {
  if (!personaA || !personaB) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-400">
        Pick a persona pair and topic to start.
      </div>
    );
  }

  const aActive = activeSpeaker === 'a';
  const bActive = activeSpeaker === 'b';
  const aSpeaking = aActive && isStreaming;
  const bSpeaking = bActive && isStreaming;

  return (
    <div className="flex-1 w-full flex flex-col items-center justify-end gap-8 px-8 pb-8">
      <div className="w-full flex items-end justify-around gap-8">
        {/* Left debater */}
        <div className="flex flex-col items-center gap-4 flex-1 max-w-md">
          <Bubble
            key={`a-${turnKeyA}`}
            text={currentTextA}
            color={personaA.color}
            streaming={aSpeaking}
            visible={bubbleVisibleA}
            side="left"
          />
          <Character
            name={personaA.name}
            color={personaA.color}
            emoji={personaA.emoji}
            side="left"
            isSpeaking={aSpeaking}
            isActive={aActive}
          />
        </div>

        {/* VS separator */}
        <div className="text-4xl font-black text-slate-600/60 pb-20 select-none">VS</div>

        {/* Right debater */}
        <div className="flex flex-col items-center gap-4 flex-1 max-w-md">
          <Bubble
            key={`b-${turnKeyB}`}
            text={currentTextB}
            color={personaB.color}
            streaming={bSpeaking}
            visible={bubbleVisibleB}
            side="right"
          />
          <Character
            name={personaB.name}
            color={personaB.color}
            emoji={personaB.emoji}
            side="right"
            isSpeaking={bSpeaking}
            isActive={bActive}
          />
        </div>
      </div>
    </div>
  );
}

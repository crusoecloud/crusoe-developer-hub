import type { Episode } from '../types';

interface Props {
  episodes: Episode[];
  activeId: string | null;
  onSelect: (ep: Episode) => void;
  onDelete: (id: string) => void;
}

export default function EpisodeLibrary({ episodes, activeId, onSelect, onDelete }: Props) {
  if (episodes.length === 0) {
    return (
      <div className="text-gray-400 text-sm text-center py-6">
        No episodes yet
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {episodes.map((ep) => (
        <div
          key={ep.id}
          className={`group flex items-center gap-2 p-2 rounded-lg cursor-pointer transition-colors ${
            activeId === ep.id
              ? 'bg-purple-600/20 border border-purple-500/30'
              : 'hover:bg-gray-100'
          }`}
          onClick={() => onSelect(ep)}
        >
          <span className="text-lg shrink-0">&#127897;</span>
          <div className="flex-1 min-w-0">
            <p className="text-sm text-gray-800 truncate">{ep.topic}</p>
            <p className="text-xs text-gray-500">
              {ep.transcript.length} lines &middot;{' '}
              {new Date(ep.createdAt).toLocaleDateString()}
            </p>
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete(ep.id);
            }}
            className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-400 transition-all text-xs"
          >
            &#10005;
          </button>
        </div>
      ))}
    </div>
  );
}

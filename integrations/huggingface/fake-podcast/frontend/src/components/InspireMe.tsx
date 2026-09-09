import { useState } from 'react';
import { fetchInspiration } from '../api';

const CATEGORIES = ['all', 'funny', 'techy', 'deep', 'business', 'weird', 'science', 'history', 'pop-culture'];

interface Props {
  onSelect: (topic: string) => void;
}

export default function InspireMe({ onSelect }: Props) {
  const [ideas, setIdeas] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [category, setCategory] = useState('all');
  const [selected, setSelected] = useState<number | null>(null);

  const generate = async () => {
    setLoading(true);
    setSelected(null);
    try {
      const data = await fetchInspiration(category);
      setIdeas(data.ideas);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-3">
      {/* Category pills */}
      <div className="flex flex-wrap gap-2">
        {CATEGORIES.map((cat) => (
          <button
            key={cat}
            onClick={() => setCategory(cat)}
            className={`px-3 py-1 rounded-full text-sm capitalize transition-colors ${
              category === cat
                ? 'bg-purple-600 text-white'
                : 'bg-gray-100 text-gray-500 hover:text-gray-800 border border-gray-300'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Inspire button */}
      <button
        onClick={generate}
        disabled={loading}
        className="btn-secondary w-full flex items-center justify-center gap-2 text-lg py-3"
      >
        {loading ? (
          <span className="animate-spin text-xl">&#9881;</span>
        ) : (
          <span className="text-xl">&#127922;</span>
        )}
        {loading ? 'Generating ideas...' : 'Inspire Me'}
      </button>

      {/* Ideas list */}
      {ideas.length > 0 && (
        <div className="space-y-2">
          {ideas.map((idea, i) => (
            <button
              key={i}
              onClick={() => {
                setSelected(i);
                onSelect(idea);
              }}
              className={`w-full text-left p-3 rounded-lg transition-all ${
                selected === i
                  ? 'bg-purple-600/20 border-purple-500 border ring-1 ring-purple-500/30'
                  : 'bg-gray-50 border border-gray-200 hover:border-gray-400'
              }`}
            >
              <span className="text-yellow-400 mr-2">&#128161;</span>
              {idea}
            </button>
          ))}

          {selected !== null && (
            <button
              onClick={() => onSelect(ideas[selected])}
              className="btn-primary w-full py-2.5"
            >
              Use This Topic &#9654;
            </button>
          )}
        </div>
      )}
    </div>
  );
}

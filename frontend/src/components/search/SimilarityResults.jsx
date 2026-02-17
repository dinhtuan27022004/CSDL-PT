import React from 'react';
import useImageStore from '../../store/useImageStore';
import Card from '../common/Card';
import Badge from '../common/Badge';
import { formatDate } from '../../utils/formatters';
import { Image as ImageIcon } from 'lucide-react';

const SimilarityResults = () => {
    const { similarityResults, similarityLoading } = useImageStore();

    if (similarityResults.length === 0 && !similarityLoading) {
        return null;
    }

    return (
        <Card className="p-6">
            <h3 className="text-xl font-semibold text-white mb-6">Search Results</h3>

            <div className="flex flex-col gap-4">
                {similarityResults.map((item, index) => (
                    <div
                        key={item.id}
                        className="bg-slate-800/50 border border-slate-700/50 rounded-lg overflow-hidden hover:border-slate-500/50 transition-all flex flex-row h-[200px] group"
                    >
                        {/* Image Preview - Left */}
                        <div className="relative w-[200px] aspect-square flex-shrink-0 bg-slate-900 flex items-center justify-center overflow-hidden">
                            {item.previewUrl ? (
                                <img
                                    src={item.previewUrl}
                                    alt={item.file_name}
                                    className="w-full h-full object-cover transition-transform group-hover:scale-105"
                                />
                            ) : (
                                <ImageIcon className="w-12 h-12 text-slate-600" />
                            )}
                            {/* Rank Badge */}
                            <div className="absolute top-2 left-2 bg-slate-900/80 text-white text-xs px-2 py-1 rounded font-mono">
                                #{index + 1}
                            </div>
                        </div>

                        {/* Metadata - Right */}
                        <div className="p-4 flex-1 flex flex-col">
                            <div className="flex items-start justify-between mb-2">
                                <div className="min-w-0">
                                    <h4 className="font-medium text-white text-base truncate" title={item.file_name}>
                                        {item.file_name}
                                    </h4>
                                    <p className="text-sm text-slate-400 font-mono">
                                        {item.width} × {item.height}
                                    </p>
                                </div>
                                <div className="flex flex-col items-end">
                                    <Badge variant={item.similarity > 80 ? 'success' : item.similarity > 50 ? 'warning' : 'default'} className="text-sm font-bold">
                                        {item.similarity.toFixed(1)}% Match
                                    </Badge>
                                </div>
                            </div>

                            {/* Key Stats Grid */}
                            <div className="grid grid-cols-4 gap-4 text-sm mt-auto">
                                <div className="flex flex-col">
                                    <span className="text-slate-500 text-xs mb-1">Uploaded</span>
                                    <span className="text-slate-300 text-xs">
                                        {formatDate(item.created_at)}
                                    </span>
                                </div>
                                <div className="flex flex-col">
                                    <span className="text-slate-500 text-xs mb-1">Color</span>
                                    <div className="flex items-center gap-2">
                                        <div
                                            className="w-3 h-3 rounded-full border border-slate-600"
                                            style={{ backgroundColor: item.dominant_color_hex }}
                                        />
                                        <span className="text-slate-300 text-xs font-mono">{item.dominant_color_hex}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </Card>
    );
};

export default SimilarityResults;

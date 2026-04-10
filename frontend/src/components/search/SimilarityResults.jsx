import React from 'react';
import useImageStore from '../../store/useImageStore';
import Card from '../common/Card';
import Badge from '../common/Badge';
import { formatDate } from '../../utils/formatters';
import { Image as ImageIcon } from 'lucide-react';

const SimilarityResults = () => {
    const { similarityResults, similarityLoading, queryImageFeatures, queryImagePreviewUrl } = useImageStore();
    
    const getSimilarityColor = (percent) => {
        // Map 0-100 to 0 (Red) - 120 (Green) in HSL
        const hue = (percent * 1.2).toFixed(0); 
        return `hsl(${hue}, 80%, 50%)`;
    };

    if (similarityResults.length === 0 && !similarityLoading) {
        return null;
    }

    return (
        <Card className="p-6">
            <h3 className="text-xl font-semibold text-white mb-6">Search Results</h3>

            {/* Display Query Image Features if available */}
            {queryImageFeatures && (
                <div className="mb-8">
                    <h4 className="text-sm font-semibold text-slate-400 mb-3 uppercase tracking-wider">Query Image Features</h4>
                    <div className="bg-slate-800/80 border border-primary-500/50 rounded-lg overflow-hidden flex flex-row h-[300px]">
                        {/* Image Preview - Left */}
                        <div className="relative w-[300px] aspect-square flex-shrink-0 bg-slate-900 flex items-center justify-center overflow-hidden">
                            {queryImagePreviewUrl ? (
                                <img
                                    src={queryImagePreviewUrl}
                                    alt="Query Image"
                                    className="w-full h-full object-cover"
                                />
                            ) : (
                                <ImageIcon className="w-12 h-12 text-slate-600" />
                            )}
                            {/* Input Badge */}
                            <div className="absolute top-2 left-2 bg-primary-500 text-white text-xs px-2 py-1 rounded font-mono shadow-lg shadow-black/50">
                                INPUT IMAGE
                            </div>
                        </div>

                        {/* Metadata - Right */}
                        <div className="p-4 flex-1 flex flex-col justify-between">
                            <div className="flex items-start">
                                <div className="flex-1 min-w-0">
                                    <h4 className="font-medium text-white text-base mb-1 truncate">
                                        Query Image
                                    </h4>
                                    <p className="text-sm text-slate-400 font-mono">
                                        {queryImageFeatures.width} × {queryImageFeatures.height}
                                    </p>
                                </div>
                            </div>

                            {/* Stats Grid */}
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm mb-2">
                                <div className="flex flex-col">
                                    <span className="text-slate-500 text-xs mb-1">Brightness</span>
                                    <span className="text-slate-200 font-medium">{(queryImageFeatures.brightness * 100).toFixed(1)}%</span>
                                </div>
                                <div className="flex flex-col">
                                    <span className="text-slate-500 text-xs mb-1">Contrast</span>
                                    <span className="text-slate-200 font-medium">{(queryImageFeatures.contrast * 100).toFixed(1)}%</span>
                                </div>
                                <div className="flex flex-col">
                                    <span className="text-slate-500 text-xs mb-1">Saturation</span>
                                    <span className="text-slate-200 font-medium">
                                        {queryImageFeatures.saturation !== undefined ? (queryImageFeatures.saturation * 100).toFixed(1) + '%' : 'N/A'}
                                    </span>
                                </div>
                                <div className="flex flex-col">
                                    <span className="text-slate-500 text-xs mb-1">Edge Density</span>
                                    <span className="text-slate-200 font-medium">{(queryImageFeatures.edge_density * 100).toFixed(1)}%</span>
                                </div>
                                <div className="flex flex-col">
                                    <span className="text-slate-500 text-xs mb-1">Dominant Color</span>
                                    <div className="flex items-center gap-2">
                                        <div
                                            className="w-4 h-4 rounded border border-slate-600"
                                            style={{ backgroundColor: queryImageFeatures.dominant_color_hex }}
                                        />
                                        <span className="text-slate-200 font-mono text-xs">{queryImageFeatures.dominant_color_hex}</span>
                                    </div>
                                </div>
                                 <div className="flex flex-col">
                                    <span className="text-slate-500 text-xs mb-1">Status</span>
                                    <span className="text-primary-400 font-medium text-xs whitespace-nowrap">
                                        Extracted for search
                                    </span>
                                </div>
                            </div>

                             {/* Raw Features JSON */}
                             {queryImageFeatures.features_json && (
                                <div className="mt-auto pt-2 border-t border-slate-700/50 flex flex-col flex-1 min-h-0">
                                    <span className="text-slate-500 text-[10px] uppercase font-bold tracking-wider mb-1">Raw Features</span>
                                    <div className="bg-slate-950/50 rounded p-2 overflow-y-auto text-[10px] font-mono text-slate-400 border border-slate-800/50">
                                        <pre className="whitespace-pre-wrap break-all">
                                            {JSON.stringify(queryImageFeatures.features_json, null, 2)}
                                        </pre>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            <div className="flex flex-col gap-4 overflow-y-auto pr-2">
                {similarityResults.map((item, index) => (
                    <div
                        key={item.id}
                        className="bg-slate-800/50 border border-slate-700/50 rounded-lg overflow-hidden hover:border-slate-500/50 transition-all flex flex-row h-[380px] group"
                    >
                        {/* Image Preview - Left (LARGER) */}
                        <div className="relative w-[450px] h-full shrink-0 bg-slate-900 flex items-center justify-center overflow-hidden border-r border-slate-700/50">
                            {item.previewUrl ? (
                                <img
                                    src={item.previewUrl}
                                    alt={item.file_name}
                                    className="w-full h-full object-cover transition-transform group-hover:scale-105"
                                />
                            ) : (
                                <ImageIcon className="w-12 h-12 text-slate-600" />
                            )}
                            
                            {/* Similarity Badge Overlay */}
                            <div className="absolute top-2 right-2">
                                <Badge variant={item.similarity > 80 ? 'success' : item.similarity > 50 ? 'warning' : 'default'} className="text-sm font-bold shadow-lg shadow-black/50">
                                    {item.similarity.toFixed(1)}% Match
                                </Badge>
                            </div>
                            
                            {/* Rank Badge overlay */}
                            <div className="absolute top-2 left-2 bg-slate-900/80 text-white text-xs px-2 py-1 rounded font-mono shadow-lg shadow-black/50">
                                #{index + 1}
                            </div>
                        </div>

                        {/* Visual Features - Middle Column (Fixed width) */}
                        <div className="w-[180px] shrink-0 p-4 border-r border-slate-700/50 flex flex-col gap-5 bg-slate-900/20">
                            <span className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1">Visual Patterns</span>
                            
                            {item.hogPreviewUrl && (
                                <div className="flex flex-col gap-1.5">
                                    <div className="flex justify-between items-center px-0.5">
                                        <span className="text-[10px] text-slate-400 font-bold">HOG (Str)</span>
                                        <span className="text-[10px] font-mono font-bold" style={{ color: getSimilarityColor(item.hog_similarity) }}>
                                            {item.hog_similarity.toFixed(0)}%
                                        </span>
                                    </div>
                                    <div className="w-full aspect-square rounded border border-slate-700 overflow-hidden bg-black shadow-inner mb-1">
                                        <img src={item.hogPreviewUrl} alt="HOG" className="w-full h-full object-contain" />
                                    </div>
                                    <div className="h-1 w-full bg-slate-950 rounded-full overflow-hidden">
                                        <div 
                                            className="h-full" 
                                            style={{ 
                                                width: `${item.hog_similarity}%`,
                                                backgroundColor: getSimilarityColor(item.hog_similarity)
                                            }} 
                                        />
                                    </div>
                                </div>
                            )}
                            
                            {item.huPreviewUrl && (
                                <div className="flex flex-col gap-1.5">
                                    <div className="flex justify-between items-center px-0.5">
                                        <span className="text-[10px] text-slate-400 font-bold">HU (Shape)</span>
                                        <span className="text-[10px] font-mono font-bold" style={{ color: getSimilarityColor(item.hu_moments_similarity) }}>
                                            {item.hu_moments_similarity.toFixed(0)}%
                                        </span>
                                    </div>
                                    <div className="w-full aspect-square rounded border border-slate-700 overflow-hidden bg-black shadow-inner mb-1">
                                        <img src={item.huPreviewUrl} alt="Hu" className="w-full h-full object-contain" />
                                    </div>
                                    <div className="h-1 w-full bg-slate-950 rounded-full overflow-hidden">
                                        <div 
                                            className="h-full" 
                                            style={{ 
                                                width: `${item.hu_moments_similarity}%`,
                                                backgroundColor: getSimilarityColor(item.hu_moments_similarity)
                                            }} 
                                        />
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Metadata & Stats - Right Column */}
                        <div className="p-5 flex-1 flex flex-col min-w-0">
                            <div className="mb-6">
                                <div className="flex justify-between items-start mb-4">
                                    <h4 className="font-bold text-white text-lg truncate flex-1 mr-4" title={item.file_name}>
                                        {item.file_name}
                                    </h4>
                                    <div className="flex flex-col items-end">
                                        <span className="text-[10px] text-slate-500 uppercase font-black tracking-widest">Total Match</span>
                                        <span 
                                            className="text-2xl font-black font-mono leading-none"
                                            style={{ color: getSimilarityColor(item.similarity) }}
                                        >
                                            {item.similarity.toFixed(1)}%
                                        </span>
                                    </div>
                                </div>
                                
                                <div className="flex flex-col gap-4">
                                    <div className="flex items-center gap-3">
                                        <span className="text-xs text-slate-400 font-mono bg-slate-900 px-2 py-0.5 rounded">
                                            {item.width} × {item.height}
                                        </span>
                                        
                                        {item.dino_similarity !== undefined && (
                                            <div className="flex items-center gap-3 flex-1">
                                                <span className="text-[10px] text-slate-500 font-bold uppercase shrink-0">Semantic (DINO)</span>
                                                <div className="flex-1 h-1.5 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                                                    <div 
                                                        className="h-full" 
                                                        style={{ 
                                                            width: `${item.dino_similarity}%`,
                                                            backgroundColor: getSimilarityColor(item.dino_similarity)
                                                        }} 
                                                    />
                                                </div>
                                                <span 
                                                    className="text-xs font-bold font-mono shrink-0" 
                                                    style={{ color: getSimilarityColor(item.dino_similarity) }}
                                                >
                                                    {item.dino_similarity.toFixed(1)}%
                                                </span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>

                            <div className="flex-1 flex flex-col gap-6">
                                <span className="text-[10px] text-slate-500 uppercase font-black tracking-widest block border-b border-slate-700/50 pb-1">Detailed Metrics</span>
                                
                                <div className="grid grid-cols-2 gap-x-8 gap-y-4">
                                    {/* Brightness */}
                                    <div className="flex flex-col">
                                        <div className="flex justify-between items-center mb-1">
                                            <span className="text-slate-500 text-[10px] font-bold uppercase tracking-tighter">Brightness</span>
                                            <span 
                                                className="text-[10px] font-bold font-mono" 
                                                style={{ color: getSimilarityColor(item.brightness_similarity) }}
                                            >
                                                {item.brightness.toFixed(2)} ({item.brightness_similarity.toFixed(1)}%)
                                            </span>
                                        </div>
                                        <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                                            <div 
                                                className="h-full transition-all duration-300" 
                                                style={{ 
                                                    width: `${item.brightness_similarity}%`,
                                                    backgroundColor: getSimilarityColor(item.brightness_similarity)
                                                }} 
                                            />
                                        </div>
                                    </div>

                                    {/* Contrast */}
                                    <div className="flex flex-col">
                                        <div className="flex justify-between items-center mb-1">
                                            <span className="text-slate-500 text-[10px] font-bold uppercase tracking-tighter">Contrast</span>
                                            <span 
                                                className="text-[10px] font-bold font-mono" 
                                                style={{ color: getSimilarityColor(item.contrast_similarity) }}
                                            >
                                                {item.contrast.toFixed(2)} ({item.contrast_similarity.toFixed(1)}%)
                                            </span>
                                        </div>
                                        <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                                            <div 
                                                className="h-full transition-all duration-300" 
                                                style={{ 
                                                    width: `${item.contrast_similarity}%`,
                                                    backgroundColor: getSimilarityColor(item.contrast_similarity)
                                                }} 
                                            />
                                        </div>
                                    </div>

                                    {/* Saturation */}
                                    <div className="flex flex-col">
                                        <div className="flex justify-between items-center mb-1">
                                            <span className="text-slate-500 text-[10px] font-bold uppercase tracking-tighter">Saturation</span>
                                            <span 
                                                className="text-[10px] font-bold font-mono" 
                                                style={{ color: getSimilarityColor(item.saturation_similarity || 0) }}
                                            >
                                                {item.saturation?.toFixed(2) || 'N/A'} ({(item.saturation_similarity || 0).toFixed(1)}%)
                                            </span>
                                        </div>
                                        <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                                            <div 
                                                className="h-full transition-all duration-300" 
                                                style={{ 
                                                    width: `${item.saturation_similarity || 0}%`,
                                                    backgroundColor: getSimilarityColor(item.saturation_similarity || 0)
                                                }} 
                                            />
                                        </div>
                                    </div>

                                    {/* Edge Density */}
                                    <div className="flex flex-col">
                                        <div className="flex justify-between items-center mb-1">
                                            <span className="text-slate-500 text-[10px] font-bold uppercase tracking-tighter">Edges</span>
                                            <span 
                                                className="text-[10px] font-bold font-mono" 
                                                style={{ color: getSimilarityColor(item.edge_density_similarity) }}
                                            >
                                                {item.edge_density.toFixed(2)} ({item.edge_density_similarity.toFixed(1)}%)
                                            </span>
                                        </div>
                                        <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                                            <div 
                                                className="h-full transition-all duration-300" 
                                                style={{ 
                                                    width: `${item.edge_density_similarity}%`,
                                                    backgroundColor: getSimilarityColor(item.edge_density_similarity)
                                                }} 
                                            />
                                        </div>
                                    </div>
                                </div>

                                <div className="pt-4 mt-auto border-t border-slate-700/50">
                                    <div className="flex flex-col gap-1.5">
                                        <div className="flex justify-between items-center px-0.5">
                                            <span className="text-[10px] text-slate-500 uppercase font-black tracking-widest">Histogram Similarity</span>
                                            <span 
                                                className="text-xs font-bold font-mono" 
                                                style={{ color: getSimilarityColor(item.histogram_similarity) }}
                                            >
                                                {item.histogram_similarity.toFixed(1)}%
                                            </span>
                                        </div>
                                        <div className="w-full h-2 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                                            <div 
                                                className="h-full transition-all duration-300" 
                                                style={{ 
                                                    width: `${item.histogram_similarity}%`,
                                                    backgroundColor: getSimilarityColor(item.histogram_similarity)
                                                }} 
                                            />
                                        </div>
                                    </div>
                                </div>
                                {/* Raw Features JSON */}
                                {item.features_json && (
                                    <div className="mt-4 pt-2 border-t border-slate-700/50 flex flex-col min-h-0">
                                        <span className="text-slate-500 text-[10px] uppercase font-bold tracking-wider mb-1">Raw Features</span>
                                        <div className="bg-slate-950/50 rounded p-2 overflow-y-auto text-[10px] font-mono text-slate-400 border border-slate-800/50 max-h-24">
                                            <pre className="whitespace-pre-wrap break-all">
                                                {JSON.stringify(item.features_json, null, 2)}
                                            </pre>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </Card>
    );
};

export default SimilarityResults;

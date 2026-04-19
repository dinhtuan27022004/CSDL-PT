import React, { useRef, useEffect } from 'react';
import useImageStore from '../../store/useImageStore';
import Card from '../common/Card';
import Badge from '../common/Badge';
import { formatDate } from '../../utils/formatters';
import { Image as ImageIcon, Sparkles, BrainCircuit } from 'lucide-react';

const ResultCard = ({ item, index, getSimilarityColor }) => {
    const ribbonRef = useRef(null);
    const { syncScrollLeft, setSyncScrollLeft } = useImageStore();
    const isScrollingRef = useRef(false);

    // Sync scroll from store to DOM
    useEffect(() => {
        if (ribbonRef.current && !isScrollingRef.current) {
            ribbonRef.current.scrollLeft = syncScrollLeft;
        }
    }, [syncScrollLeft]);

    const handleScroll = (e) => {
        if (ribbonRef.current) {
            isScrollingRef.current = true;
            setSyncScrollLeft(e.target.scrollLeft);
            // Reset flag after a short delay to allow store update to propogate
            setTimeout(() => {
                isScrollingRef.current = false;
            }, 50);
        }
    };

    return (
        <div
            className="bg-slate-800/50 border border-slate-700/50 rounded-lg overflow-hidden hover:border-slate-500/50 transition-all flex flex-row h-[480px] group"
        >
            {/* Column 1: Main Image Preview (400x400) */}
            <div className="relative w-[400px] h-full shrink-0 bg-slate-900 flex items-center justify-center overflow-hidden border-r border-slate-700/50">
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

            {/* Column 2: Stats & Metadata (220px Fixed) */}
            <div className="w-[220px] shrink-0 border-r border-slate-700/50 p-4 flex flex-col bg-slate-800/20 overflow-y-auto custom-scrollbar">
                <div className="mb-4">
                    <h4 className="font-bold text-white text-sm truncate mb-1" title={item.file_name}>
                        {item.file_name}
                    </h4>
                    <div className="flex flex-col">
                        <span className="text-[9px] text-slate-500 uppercase font-black tracking-widest">Total Match</span>
                        <span
                            className="text-xl font-black font-mono leading-none"
                            style={{ color: getSimilarityColor(item.similarity) }}
                        >
                            {item.similarity.toFixed(1)}%
                        </span>
                    </div>
                </div>

                <div className="space-y-3 flex-1">
                    <div className="space-y-2">
                        <div className="flex justify-between items-center">
                            <span className="text-slate-500 text-[10px] font-bold uppercase">Brightness</span>
                            <span className="text-[10px] text-slate-300 font-mono">{(item.brightness_similarity || 0).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-slate-500 text-[10px] font-bold uppercase">Contrast</span>
                            <span className="text-[10px] text-slate-300 font-mono">{(item.contrast_similarity || 0).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-slate-500 text-[10px] font-bold uppercase">Saturation</span>
                            <span className="text-[10px] text-slate-300 font-mono">{(item.saturation_similarity || 0).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-slate-500 text-[10px] font-bold uppercase">Edges</span>
                            <span className="text-[10px] text-slate-300 font-mono">{(item.edge_density_similarity || 0).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-slate-500 text-[10px] font-bold uppercase">Fourier</span>
                            <span className="text-[10px] text-slate-300 font-mono">{(item.zernike_similarity || 0).toFixed(0)}%</span>
                        </div>
                    </div>

                    <div className="pt-2 border-t border-indigo-500/30 space-y-2">
                        <div className="flex justify-between items-center">
                            <span className="text-indigo-400 text-[10px] font-bold uppercase">Embedding Match</span>
                            <span className="text-[10px] font-bold font-mono text-indigo-300">{(item.semantic_similarity || 0).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-emerald-400 text-[10px] font-bold uppercase">Entity Match</span>
                            <span className="text-[10px] font-bold font-mono text-emerald-300">{(item.entity_similarity || 0).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-amber-400 text-[10px] font-bold uppercase">Category Match</span>
                            <span className="text-[10px] font-bold font-mono text-amber-300">{(item.category_similarity || 0).toFixed(0)}%</span>
                        </div>
                    </div>

                    <div className="pt-2 border-t border-slate-700/50 space-y-2">
                        <div className="flex justify-between items-center">
                            <span className="text-slate-500 text-[10px] font-bold uppercase">HSV Hist</span>
                            <span className="text-[10px] font-bold font-mono text-primary-400">{(item.hsv_histogram_similarity || 0).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-slate-500 text-[10px] font-bold uppercase">RGB Hist</span>
                            <span className="text-[10px] font-bold font-mono text-primary-400">{(item.rgb_histogram_similarity || 0).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-slate-500 text-[10px] font-bold uppercase">HSV CDF</span>
                            <span className="text-[10px] font-bold font-mono text-emerald-400">{(item.hsv_cdf_similarity || 0).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-slate-500 text-[10px] font-bold uppercase">RGB CDF</span>
                            <span className="text-[10px] font-bold font-mono text-emerald-400">{(item.rgb_cdf_similarity || 0).toFixed(0)}%</span>
                        </div>
                    </div>

                    <div className="pt-2 border-t border-slate-700/50">
                        <span className="text-slate-500 text-[10px] font-bold uppercase block mb-1">Dominant Color</span>
                        <div className="flex items-center gap-2">
                            <div
                                className="w-5 h-5 rounded border border-slate-600"
                                style={{ backgroundColor: item.dominant_color_hex }}
                            />
                            <span className="text-slate-400 font-mono text-[10px]">{item.dominant_color_hex}</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Column 3: AI Insights (300px Fixed) - NEW */}
            <div className="w-[300px] shrink-0 border-r border-slate-700/50 p-5 flex flex-col bg-blue-900/10 overflow-y-auto custom-scrollbar">
                <div className="flex items-center gap-2 mb-4">
                    <BrainCircuit className="w-4 h-4 text-primary-400" />
                    <span className="text-[10px] text-primary-400 uppercase font-black tracking-widest">AI Insights</span>
                </div>

                <div className="mb-6">
                    <span className="text-[9px] text-slate-500 uppercase font-bold block mb-1.5">Detected Category</span>
                    <div className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold bg-primary-500/20 text-primary-400 border border-primary-500/30">
                        {item.category || 'Analyzing...'}
                    </div>
                </div>

                <div className="mb-4 flex flex-col min-h-0">
                    <span className="text-[9px] text-slate-500 uppercase font-bold block mb-2">Detected Objects</span>
                    <div className="flex flex-wrap gap-1.5 overflow-y-auto max-h-[80px] custom-scrollbar pr-1">
                        {item.entities && item.entities.length > 0 ? (
                            item.entities.map((tag, idx) => (
                                <span 
                                    key={idx} 
                                    className="px-2 py-0.5 rounded bg-slate-700/50 text-[10px] text-slate-300 border border-slate-600/50 hover:border-primary-500/50 transition-colors"
                                >
                                    {tag}
                                </span>
                            ))
                        ) : (
                            <span className="text-[10px] text-slate-600 italic">None detected</span>
                        )}
                    </div>
                </div>

                <div className="flex-1 flex flex-col min-h-0">
                    <span className="text-[9px] text-slate-500 uppercase font-bold block mb-2">Visual Description</span>
                    <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700/50 flex-1 overflow-y-auto">
                        <p className="text-sm text-slate-300 leading-relaxed italic">
                            {item.description || "The system is extracting advanced semantic details for this image..."}
                        </p>
                    </div>
                </div>
            </div>

            {/* Column 4: Visual Ribbon (Horizontal Scroll) */}
            <div 
                ref={ribbonRef}
                onScroll={handleScroll}
                className="flex-1 overflow-x-auto overflow-y-hidden bg-slate-900/40 custom-scrollbar flex flex-row p-4 gap-8 scroll-smooth"
            >
                {item.histogramPreviewUrl && (
                    <div className="flex flex-col gap-2 shrink-0 w-[600px]">
                        <span className="text-[10px] text-slate-500 uppercase font-black tracking-widest px-1">Histogram Distribution (HSV + RGB Channels)</span>
                        <div className="flex-1 rounded border border-slate-700 overflow-hidden bg-slate-950 shadow-2xl">
                            <img src={item.histogramPreviewUrl} alt="Histogram" className="w-full h-full object-fill" />
                        </div>
                    </div>
                )}

                {item.hogPreviewUrl && (
                    <div className="flex flex-col gap-2 shrink-0 w-[400px]">
                        <span className="text-[10px] text-slate-500 uppercase font-black tracking-widest px-1">HOG Gradient Map ({(item.hog_similarity || 0).toFixed(0)}%)</span>
                        <div className="flex-1 rounded border border-slate-700 overflow-hidden bg-black shadow-2xl">
                            <img src={item.hogPreviewUrl} alt="HOG" className="w-full h-full object-contain" />
                        </div>
                    </div>
                )}

                {item.ccvPreviewUrl && (
                    <div className="flex flex-col gap-2 shrink-0 w-[400px]">
                        <span className="text-[10px] text-slate-500 uppercase font-black tracking-widest px-1">Color Coherence ({(item.ccv_similarity || 0).toFixed(0)}%)</span>
                        <div className="flex-1 rounded border border-slate-700 overflow-hidden bg-black shadow-2xl">
                            <img src={item.ccvPreviewUrl} alt="CCV" className="w-full h-full object-contain" />
                        </div>
                    </div>
                )}

                {item.gaborPreviewUrl && (
                    <div className="flex flex-col gap-2 shrink-0 w-[400px]">
                        <span className="text-[10px] text-slate-500 uppercase font-black tracking-widest px-1">Gabor Texture ({(item.gabor_similarity || 0).toFixed(0)}%)</span>
                        <div className="flex-1 rounded border border-slate-700 overflow-hidden bg-black shadow-2xl">
                            <img src={item.gaborPreviewUrl} alt="Gabor" className="w-full h-full object-contain" />
                        </div>
                    </div>
                )}

                {item.lbpPreviewUrl && (
                    <div className="flex flex-col gap-2 shrink-0 w-[400px]">
                        <span className="text-[10px] text-slate-500 uppercase font-black tracking-widest px-1">LBP Texture ({(item.lbp_similarity || 0).toFixed(0)}%)</span>
                        <div className="flex-1 rounded border border-slate-700 overflow-hidden bg-black shadow-2xl">
                            <img src={item.lbpPreviewUrl} alt="LBP" className="w-full h-full object-contain" />
                        </div>
                    </div>
                )}

                {item.cellColorPreviewUrl && (
                    <div className="flex flex-col gap-2 shrink-0 w-[400px]">
                        <span className="text-[10px] text-slate-500 uppercase font-black tracking-widest px-1">Cell Color Grid ({(item.cell_color_similarity || 0).toFixed(0)}%)</span>
                        <div className="flex-1 rounded border border-slate-700 overflow-hidden bg-black shadow-2xl">
                            <img src={item.cellColorPreviewUrl} alt="Cell Color" className="w-full h-full object-contain" />
                        </div>
                    </div>
                )}

                {item.huPreviewUrl && (
                    <div className="flex flex-col gap-2 shrink-0 w-[400px]">
                        <span className="text-[10px] text-slate-500 uppercase font-black tracking-widest px-1">Hu Moments ({(item.hu_moments_similarity || 0).toFixed(0)}%)</span>
                        <div className="flex-1 rounded border border-slate-700 overflow-hidden bg-black shadow-2xl">
                            <img src={item.huPreviewUrl} alt="Hu" className="w-full h-full object-contain" />
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

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

                            {/* Semantic Info - NEW */}
                            {(queryImageFeatures.category || queryImageFeatures.description) && (
                                <div className="mt-4 p-3 bg-blue-900/10 rounded-lg border border-primary-500/20">
                                    <div className="flex items-center gap-2 mb-2">
                                        <BrainCircuit className="w-3 h-3 text-primary-400" />
                                        <span className="text-[10px] text-primary-400 uppercase font-black tracking-widest">AI Result</span>
                                    </div>
                                    <div className="flex flex-col gap-2">
                                        <div className="inline-flex px-2 py-0.5 rounded-full text-[10px] font-bold bg-primary-500/20 text-primary-400 self-start">
                                            {queryImageFeatures.category || 'Analyzing...'}
                                        </div>
                                        <p className="text-xs text-slate-300 italic line-clamp-2">
                                            {queryImageFeatures.description}
                                        </p>
                                    </div>
                                </div>
                            )}

                            {/* Raw Features JSON */}
                            {queryImageFeatures.features_json && (
                                <div className="mt-auto pt-4 border-t border-slate-700/50 flex flex-col flex-1 min-h-0">
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

            <div className="flex flex-col gap-6 overflow-y-auto pr-2">
                {similarityResults.map((item, index) => (
                    <ResultCard 
                        key={item.id} 
                        item={item} 
                        index={index} 
                        getSimilarityColor={getSimilarityColor} 
                    />
                ))}
            </div>
        </Card>
    );
};

export default SimilarityResults;

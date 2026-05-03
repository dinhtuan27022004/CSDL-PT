import React, { useRef, useEffect } from 'react';
import useImageStore from '../../store/useImageStore';
import Card from '../common/Card';
import Badge from '../common/Badge';
import { formatDate } from '../../utils/formatters';
import { Image as ImageIcon, Sparkles, BrainCircuit, Zap } from 'lucide-react';

const VIS_COMPONENTS = [
    { id: 'histogram', label: 'Histogram', width: 450, color: 'text-primary-400' },
    { id: 'hog', label: 'HOG Map', width: 180, color: 'text-orange-400' },
    { id: 'ccv', label: 'Color Coherence', width: 180, color: 'text-emerald-400' },
    { id: 'gabor', label: 'Gabor Texture', width: 180, color: 'text-purple-400' },
    { id: 'lbp', label: 'LBP Texture', width: 180, color: 'text-pink-400' },
    { id: 'cellColor', label: 'Cell Grid', width: 180, color: 'text-lime-400' },
    { id: 'hu', label: 'Hu Moments', width: 180, color: 'text-amber-400' },
];

const SimilarityResults = () => {
    const { similarityResults, gtResults, similarityLoading, queryImageFeatures, queryImagePreviewUrl } = useImageStore();
    const [enabledVis, setEnabledVis] = React.useState({
        histogram: true,
        hog: true,
        ccv: false,
        gabor: false,
        lbp: true,
        cellColor: false,
        hu: false
    });

    const toggleVis = (id) => {
        setEnabledVis(prev => ({ ...prev, [id]: !prev[id] }));
    };

    const bgrToHex = (bgr) => {
        if (!bgr || !Array.isArray(bgr) || bgr.length < 3) return '#808080';
        const [b, g, r] = bgr;
        return '#' + [r, g, b].map(x => {
            const hex = Math.round(x).toString(16);
            return hex.length === 1 ? '0' + hex : hex;
        }).join('');
    };

    if (similarityResults.length === 0 && !similarityLoading) {
        return null;
    }

    return (
        <Card className="p-4 bg-slate-950 border-slate-800">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 mb-4">
                <div>
                    <h3 className="text-xl font-black text-white uppercase tracking-tighter">Search Analysis Matrix</h3>
                    <p className="text-slate-500 text-xs">Professional vision analysis matrix • 1:1 Vis Mode • Height Synchronized</p>
                </div>
            </div>

            {/* Display Query Image Features */}
            {queryImageFeatures && (
                <div className="mb-6">
                    <div className="bg-slate-900 border border-slate-700 overflow-hidden flex flex-col md:flex-row shadow-2xl">
                        <div className="relative w-full md:w-[450px] aspect-video bg-black flex items-center justify-center overflow-hidden border-b md:border-b-0 md:border-r border-slate-700/50">
                            {queryImagePreviewUrl ? (
                                <img
                                    src={queryImagePreviewUrl}
                                    alt="Query Image"
                                    className="w-full h-full object-contain"
                                />
                            ) : (
                                <ImageIcon className="w-10 h-10 text-slate-800" />
                            )}
                            <div className="absolute top-2 left-2 bg-primary-600 text-white text-[9px] font-black px-2 py-0.5 shadow-xl uppercase tracking-widest">
                                Input Source
                            </div>
                        </div>

                        <div className="p-4 flex-1 flex flex-col bg-slate-900/50">
                            <div className="flex justify-between items-start mb-4">
                                <div>
                                    <h4 className="text-sm font-black text-white uppercase tracking-widest mb-0.5">Query Metadata</h4>
                                    <span className="text-[10px] font-mono text-slate-500">{queryImageFeatures.width}px × {queryImageFeatures.height}px</span>
                                </div>
                                <BrainCircuit className="w-6 h-6 text-primary-500/30" />
                            </div>

                            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                                <div className="space-y-0.5">
                                    <span className="text-[9px] font-black text-slate-600 uppercase">Brightness</span>
                                    <div className="h-1 w-full bg-slate-800 overflow-hidden">
                                        <div className="h-full bg-blue-500" style={{ width: `${(queryImageFeatures.brightness || 0) * 100}%` }}></div>
                                    </div>
                                    <span className="text-[10px] font-mono text-slate-400">{((queryImageFeatures.brightness || 0) * 100).toFixed(1)}%</span>
                                </div>
                                <div className="space-y-0.5">
                                    <span className="text-[9px] font-black text-slate-600 uppercase">Contrast</span>
                                    <div className="h-1 w-full bg-slate-800 overflow-hidden">
                                        <div className="h-full bg-emerald-500" style={{ width: `${(queryImageFeatures.contrast || 0) * 100}%` }}></div>
                                    </div>
                                    <span className="text-[10px] font-mono text-slate-400">{((queryImageFeatures.contrast || 0) * 100).toFixed(1)}%</span>
                                </div>
                                <div className="space-y-0.5">
                                    <span className="text-[9px] font-black text-slate-600 uppercase">Edges</span>
                                    <div className="h-1 w-full bg-slate-800 overflow-hidden">
                                        <div className="h-full bg-orange-500" style={{ width: `${(queryImageFeatures.edge_density || 0) * 100}%` }}></div>
                                    </div>
                                    <span className="text-[10px] font-mono text-slate-400">{((queryImageFeatures.edge_density || 0) * 100).toFixed(1)}%</span>
                                </div>
                                <div className="space-y-0.5">
                                    <span className="text-[9px] font-black text-slate-600 uppercase">Main Color</span>
                                    <div className="flex items-center gap-2 mt-1">
                                        <div className="w-6 h-3 border border-slate-700" style={{ backgroundColor: bgrToHex(queryImageFeatures.dominant_color_vector) }}></div>
                                        <span className="text-[10px] font-mono text-slate-400 uppercase">{bgrToHex(queryImageFeatures.dominant_color_vector)}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Matrix View Toggles */}
            <div className="mb-4 p-2 bg-slate-900 border border-slate-800">
                <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[9px] font-black text-slate-600 uppercase tracking-widest mr-2">Visualizers:</span>
                    {VIS_COMPONENTS.map(vis => (
                        <button
                            key={vis.id}
                            onClick={() => toggleVis(vis.id)}
                            className={`px-2 py-1 text-[9px] font-black transition-all uppercase tracking-tighter border ${
                                enabledVis[vis.id] 
                                    ? 'bg-primary-700 border-primary-600 text-white' 
                                    : 'bg-slate-800 border-slate-700 text-slate-600 hover:text-slate-500'
                            }`}
                        >
                            {vis.label}
                        </button>
                    ))}
                </div>
            </div>

            {gtResults && gtResults.length > 0 ? (
                /* Comparison Mode Layout */
                <div className="grid grid-cols-2 gap-4">
                    <div className="border-b border-primary-500/30 pb-1 mb-2">
                        <h4 className="text-sm font-black text-white uppercase tracking-tighter">Optimized Results</h4>
                    </div>
                    <div className="border-b border-emerald-500/30 pb-1 mb-2">
                        <h4 className="text-sm font-black text-white uppercase tracking-tighter">Ground Truth Results</h4>
                    </div>

                    {Array.from({ length: Math.max(similarityResults.length, gtResults.length) }).map((_, idx) => (
                        <React.Fragment key={idx}>
                            <div className="relative group">
                                {similarityResults[idx] && (
                                    <div className="bg-slate-900 border border-slate-800 overflow-hidden shadow-xl">
                                        <div className="relative w-full aspect-video bg-black">
                                            <img src={similarityResults[idx].previewUrl} className="w-full h-full object-cover" />
                                            <div className="absolute top-1 right-1 bg-primary-600/90 text-white text-[9px] font-black px-1.5 py-0.5 shadow-lg">
                                                {similarityResults[idx].similarity.toFixed(1)}%
                                            </div>
                                        </div>
                                    </div>
                                )}
                                <div className="absolute -left-3 top-1/2 -translate-y-1/2 text-[8px] font-black text-slate-700 rotate-180 [writing-mode:vertical-lr]">RANK #{idx + 1}</div>
                            </div>
                            <div className="relative group">
                                {gtResults[idx] && (
                                    <div className="bg-slate-900 border border-slate-800 overflow-hidden shadow-xl">
                                        <div className="relative w-full aspect-video bg-black">
                                            <img src={gtResults[idx].previewUrl} className="w-full h-full object-cover" />
                                            <div className="absolute top-1 right-1 bg-emerald-600/90 text-white text-[9px] font-black px-1.5 py-0.5 shadow-lg">
                                                {gtResults[idx].similarity.toFixed(1)}%
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </React.Fragment>
                    ))}
                </div>
            ) : (
                /* Matrix Table Layout */
                <div className="relative overflow-x-auto overflow-y-hidden border border-slate-800 bg-slate-950 custom-scrollbar">
                    <table className="w-full text-left border-collapse min-w-max">
                        <thead>
                            <tr className="bg-slate-900 border-b border-slate-800">
                                <th className="px-0.5 py-2 w-12 text-[9px] font-black text-slate-600 uppercase tracking-widest text-center">Rank</th>
                                <th className="px-0.5 py-2 w-80 text-[9px] font-black text-slate-600 uppercase tracking-widest">Main Result (16:9)</th>
                                <th className="px-0.5 py-2 w-80 text-[9px] font-black text-slate-600 uppercase tracking-widest">Vision Metrics</th>
                                {VIS_COMPONENTS.map(vis => enabledVis[vis.id] && (
                                    <th key={vis.id} className={`px-0.5 py-2 text-[9px] font-black uppercase tracking-widest ${vis.color}`} style={{ width: vis.width }}>
                                        {vis.label}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/50">
                            {similarityResults.map((item, idx) => (
                                <tr key={item.id} className="group hover:bg-slate-900/60 transition-colors border-b border-slate-800/30">
                                    <td className="px-0 py-1 text-center border-r border-slate-800/30">
                                        <span className="text-sm font-black text-slate-700 group-hover:text-primary-500 transition-colors">#{idx + 1}</span>
                                    </td>
                                    <td className="px-0 py-1 border-r border-slate-800/30">
                                        <div className="relative w-80 h-[180px] overflow-hidden border-x border-slate-800/50 group-hover:border-primary-500 transition-all shadow-xl">
                                            <img src={item.previewUrl} className="w-full h-full object-cover transition-transform group-hover:scale-105" />
                                            <div className="absolute top-0 right-0 bg-primary-600 text-white text-[8px] font-black px-1.5 py-0.5">
                                                {item.similarity.toFixed(1)}%
                                            </div>
                                        </div>
                                        <div className="mt-0.5 px-2 text-[7px] font-bold text-slate-600 truncate max-w-[320px] uppercase">{item.file_name}</div>
                                    </td>
                                    <td className="px-1 py-1 align-top border-r border-slate-800/30 bg-slate-900/20">
                                        <div className="grid grid-cols-2 gap-x-2 gap-y-0 text-[7px]">
                                            {/* Deep Semantic Models */}
                                            <div className="flex justify-between items-center border-b border-slate-800/50 py-0.5"><span className="text-purple-500 font-black uppercase">CLIP</span><span className="font-mono text-purple-400">{(item.clip_similarity || 0).toFixed(0)}%</span></div>
                                            <div className="flex justify-between items-center border-b border-slate-800/50 py-0.5"><span className="text-pink-500 font-black uppercase">DINO</span><span className="font-mono text-pink-400">{(item.dinov2_similarity || 0).toFixed(0)}%</span></div>
                                            <div className="flex justify-between items-center border-b border-slate-800/50 py-0.5"><span className="text-rose-500 font-black uppercase">SAM</span><span className="font-mono text-rose-400">{(item.sam_similarity || 0).toFixed(0)}%</span></div>
                                            <div className="flex justify-between items-center border-b border-slate-800/50 py-0.5"><span className="text-sky-500 font-black uppercase">DREAM</span><span className="font-mono text-sky-400">{(item.dreamsim_similarity || 0).toFixed(0)}%</span></div>
                                            
                                            {/* Multi-Space Matrix (Grouped) */}
                                            {["rgb", "hsv", "lab", "ycrcb", "hls", "xyz", "gray"].map(space => (
                                                <div key={space} className="flex justify-between items-center border-b border-slate-800/50 py-0.5">
                                                    <span className={`font-black uppercase ${
                                                        space === 'rgb' ? 'text-red-500' :
                                                        space === 'hsv' ? 'text-orange-500' :
                                                        space === 'lab' ? 'text-emerald-500' :
                                                        space === 'ycrcb' ? 'text-blue-500' :
                                                        space === 'hls' ? 'text-purple-500' :
                                                        space === 'xyz' ? 'text-amber-500' : 'text-slate-400'
                                                    }`}>{space}</span>
                                                    <span className="font-mono text-slate-400">
                                                        {(item[`${space}_hist_gauss_similarity`] || item[`${space}_hist_std_similarity`] || 0).toFixed(0)}%
                                                    </span>
                                                </div>
                                            ))}
                                            
                                            {/* Metadata */}
                                            <div className="flex justify-between items-center py-0.5 col-span-2 text-slate-500 font-bold uppercase italic">
                                                <span>{item.category || "General"}</span>
                                                <span>{(item.edge_density_similarity || 0).toFixed(0)}% Edge</span>
                                            </div>
                                        </div>
                                    </td>
                                    {enabledVis.histogram && (
                                        <td className="px-0 py-1 border-r border-slate-800/30">
                                            <div className="w-[450px] h-[180px] bg-slate-950 border-x border-slate-800 overflow-hidden">
                                                {item.histogramPreviewUrl ? <img src={item.histogramPreviewUrl} className="w-full h-full object-fill" /> : <div className="w-full h-full flex items-center justify-center text-[9px] text-slate-800 uppercase">No Data</div>}
                                            </div>
                                        </td>
                                    )}
                                    {enabledVis.hog && (
                                        <td className="px-0 py-1 border-r border-slate-800/30">
                                            <div className="w-[180px] h-[180px] bg-black border-x border-slate-800 overflow-hidden">
                                                {item.hogPreviewUrl ? <img src={item.hogPreviewUrl} className="w-full h-full object-cover" /> : <div className="w-full h-full flex items-center justify-center text-[9px] text-slate-800 uppercase">No Data</div>}
                                            </div>
                                        </td>
                                    )}
                                    {enabledVis.ccv && (
                                        <td className="px-0 py-1 border-r border-slate-800/30">
                                            <div className="w-[180px] h-[180px] bg-black border-x border-slate-800 overflow-hidden">
                                                {item.ccvPreviewUrl ? <img src={item.ccvPreviewUrl} className="w-full h-full object-cover" /> : <div className="w-full h-full flex items-center justify-center text-[9px] text-slate-800 uppercase">No Data</div>}
                                            </div>
                                        </td>
                                    )}
                                    {enabledVis.gabor && (
                                        <td className="px-0 py-1 border-r border-slate-800/30">
                                            <div className="w-[180px] h-[180px] bg-black border-x border-slate-800 overflow-hidden">
                                                {item.gaborPreviewUrl ? <img src={item.gaborPreviewUrl} className="w-full h-full object-cover" /> : <div className="w-full h-full flex items-center justify-center text-[9px] text-slate-800 uppercase">No Data</div>}
                                            </div>
                                        </td>
                                    )}
                                    {enabledVis.lbp && (
                                        <td className="px-0 py-1 border-r border-slate-800/30">
                                            <div className="w-[180px] h-[180px] bg-black border-x border-slate-800 overflow-hidden">
                                                {item.lbpPreviewUrl ? <img src={item.lbpPreviewUrl} className="w-full h-full object-cover" /> : <div className="w-full h-full flex items-center justify-center text-[9px] text-slate-800 uppercase">No Data</div>}
                                            </div>
                                        </td>
                                    )}
                                    {enabledVis.cellColor && (
                                        <td className="px-0 py-1 border-r border-slate-800/30">
                                            <div className="w-[180px] h-[180px] bg-black border-x border-slate-800 overflow-hidden">
                                                {item.cellColorPreviewUrl ? <img src={item.cellColorPreviewUrl} className="w-full h-full object-cover" /> : <div className="w-full h-full flex items-center justify-center text-[9px] text-slate-800 uppercase">No Data</div>}
                                            </div>
                                        </td>
                                    )}
                                    {enabledVis.hu && (
                                        <td className="px-0 py-1">
                                            <div className="w-[180px] h-[180px] bg-black border-l border-slate-800 overflow-hidden">
                                                {item.huPreviewUrl ? <img src={item.huPreviewUrl} className="w-full h-full object-cover" /> : <div className="w-full h-full flex items-center justify-center text-[9px] text-slate-800 uppercase">No Data</div>}
                                            </div>
                                        </td>
                                    )}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </Card>
    );
};

export default SimilarityResults;

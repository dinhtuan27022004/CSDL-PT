import React, { useRef, useEffect } from 'react';
import useImageStore from '../../store/useImageStore';
import Card from '../common/Card';
import Badge from '../common/Badge';
import { formatDate } from '../../utils/formatters';
import { Image as ImageIcon, Sparkles, BrainCircuit, Zap } from 'lucide-react';

const VIS_COMPONENTS = [
    { id: 'histogram', label: 'Histograms (21)', width: 600, color: 'text-primary-400' },
    { id: 'hog', label: 'HOG Map', width: 180, color: 'text-orange-400' },
    { id: 'ccv', label: 'Color Coherence', width: 180, color: 'text-emerald-400' },
    { id: 'gabor', label: 'Gabor Texture', width: 180, color: 'text-purple-400' },
    { id: 'lbp', label: 'LBP Texture', width: 180, color: 'text-pink-400' },
    { id: 'cellColor', label: 'Cell Grid', width: 180, color: 'text-lime-400' },
    { id: 'hu', label: 'Hu Moments', width: 180, color: 'text-amber-400' },
];

const SimilarityResults = () => {
    const { similarityResults, similarityLoading, queryImageFeatures, queryImagePreviewUrl } = useImageStore();
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

    const renderHistogramGrid = (item) => {
        const urlData = item.histogramPreviewUrl;
        if (!urlData) return <div className="w-full h-full flex items-center justify-center text-[9px] text-slate-800 uppercase">No Data</div>;
        
        try {
            // Check if it's a JSON string (multiple images) or a single URL
            const paths = typeof urlData === 'string' && urlData.startsWith('{') 
                ? JSON.parse(urlData) 
                : null;
            
            if (!paths) {
                return <img src={urlData} className="w-full h-full object-fill" />;
            }

            const spaces = ["rgb", "hsv", "lab", "ycrcb", "hls", "xyz", "gray"];
            const methods = ["std", "interp", "gauss"];

            return (
                <div className="grid grid-cols-3 gap-0.5 p-0.5 bg-slate-900 w-full h-full overflow-y-auto custom-scrollbar">
                    {spaces.map(space => (
                        methods.map(method => {
                            const key = `${space}_${method}`;
                            const path = paths[key];
                            // Map to similarity key: space_hist_method_similarity
                            const simKey = `${space}_hist_${method}_similarity`;
                            const sim = item[simKey] || item[`${space}_${method}_similarity`] || 0;
                            
                            return (
                                <div key={key} className="relative aspect-[4/3] bg-black border border-slate-800 group/hist overflow-hidden">
                                    {path ? (
                                        <img 
                                            src={path} 
                                            alt={key} 
                                            className="w-full h-full object-cover opacity-80 group-hover/hist:opacity-100 transition-opacity" 
                                        />
                                    ) : (
                                        <div className="w-full h-full flex items-center justify-center text-[6px] text-slate-700">N/A</div>
                                    )}
                                    <div className="absolute top-0 left-0 right-0 bg-black/60 backdrop-blur-[2px] py-0.5 px-1.5 flex justify-between items-center border-b border-white/5 opacity-0 group-hover/hist:opacity-100 transition-opacity">
                                        <span className="text-[5px] font-black text-slate-300 uppercase tracking-tighter truncate block">{space} {method}</span>
                                        <span className="text-[6px] font-mono font-black text-primary-400">{(sim * 100).toFixed(0)}%</span>
                                    </div>
                                    {/* Small bottom similarity badge */}
                                    <div className="absolute bottom-0.5 right-0.5 bg-black/80 px-1 rounded-[1px] text-[5px] font-mono font-black text-white/40 group-hover/hist:text-primary-400 transition-colors pointer-events-none">
                                        {(sim * 100).toFixed(0)}%
                                    </div>
                                </div>
                            );
                        })
                    ))}
                </div>
            );
        } catch (e) {
            console.error("Failed to parse histogram paths", e);
            return <img src={urlData} className="w-full h-full object-fill" />;
        }
    };

    const VisWrapper = ({ children, label, similarity }) => (
        <div className="relative w-full h-full group/vis">
            {children}
            <div className="absolute top-0 left-0 right-0 bg-slate-950/80 backdrop-blur-md px-2 py-1 flex justify-between items-center border-b border-white/5 translate-y-[-100%] group-hover/vis:translate-y-0 transition-transform duration-300">
                <span className="text-[8px] font-black text-slate-400 uppercase tracking-widest">{label}</span>
                <div className="flex items-center gap-1.5">
                    <div className="h-1 w-12 bg-slate-800 rounded-full overflow-hidden hidden sm:block">
                        <div className="h-full bg-primary-500" style={{ width: `${(similarity || 0) * 100}%` }}></div>
                    </div>
                    <span className="text-[9px] font-mono font-black text-primary-400">{((similarity || 0) * 100).toFixed(0)}%</span>
                </div>
            </div>
            {/* Permanent Mini Tag */}
            <div className="absolute bottom-1 right-1 bg-black/80 px-1.5 py-0.5 rounded text-[7px] font-mono font-black text-white/40 group-hover/vis:text-primary-400 transition-colors pointer-events-none">
                {((similarity || 0) * 100).toFixed(0)}%
            </div>
        </div>
    );

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

            {/* Matrix Table Layout */}
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
                                    <td className="px-1 py-1 align-top border-r border-slate-800/30 bg-slate-900/40" style={{ width: 450 }}>
                                        <div className="flex flex-col h-[180px] overflow-y-auto custom-scrollbar pr-1 bg-slate-950/50 p-1 border border-slate-800/50">
                                            {/* SECTION 1: SEMANTIC INTELLIGENCE */}
                                            <div className="mb-3">
                                                <div className="flex items-center gap-1 mb-1 border-b border-primary-500/20 pb-0.5">
                                                    <Sparkles className="w-2.5 h-2.5 text-primary-400" />
                                                    <span className="text-[7px] font-black text-primary-400 uppercase tracking-widest">Semantic & Context</span>
                                                </div>
                                                <div className="space-y-1">
                                                    <div className="flex flex-col gap-0.5">
                                                        <div className="flex justify-between text-[7px]">
                                                            <span className="text-slate-500">Category: <span className="text-primary-300">{item.category || "N/A"}</span></span>
                                                            <span className="font-mono text-primary-400">{((item.category_similarity || 0) * 100).toFixed(0)}%</span>
                                                        </div>
                                                        <div className="h-0.5 w-full bg-slate-800"><div className="h-full bg-primary-600" style={{ width: `${(item.category_similarity || 0) * 100}%` }}></div></div>
                                                    </div>
                                                    <div className="flex flex-col gap-0.5">
                                                        <div className="flex justify-between text-[7px]">
                                                            <span className="text-slate-500">Entities: <span className="text-emerald-400">{(item.entities || []).join(', ') || "None"}</span></span>
                                                            <span className="font-mono text-emerald-400">{((item.entity_similarity || 0) * 100).toFixed(0)}%</span>
                                                        </div>
                                                        <div className="h-0.5 w-full bg-slate-800"><div className="h-full bg-emerald-600" style={{ width: `${(item.entity_similarity || 0) * 100}%` }}></div></div>
                                                    </div>
                                                    <div className="text-[6px] text-slate-400 leading-tight bg-slate-900/50 p-1 border-l border-primary-500/50 italic">
                                                        "{item.description || "No semantic description available."}"
                                                    </div>
                                                </div>
                                            </div>


                                            {/* SECTION 3: PHYSICAL METADATA */}
                                            <div className="mb-3">
                                                <div className="flex items-center gap-1 mb-1 border-b border-slate-500/20 pb-0.5">
                                                    <Zap className="w-2.5 h-2.5 text-slate-400" />
                                                    <span className="text-[7px] font-black text-slate-400 uppercase tracking-widest">Physical & Optical</span>
                                                </div>
                                                <div className="grid grid-cols-2 gap-x-2 gap-y-1">
                                                    {[
                                                        { label: 'Brightness', key: 'brightness' },
                                                        { label: 'Contrast', key: 'contrast' },
                                                        { label: 'Saturation', key: 'saturation' },
                                                        { label: 'Edge Density', key: 'edge_density' },
                                                        { label: 'Sharpness', key: 'sharpness' }
                                                    ].map(m => (
                                                        <div key={m.key} className="flex flex-col gap-0.5">
                                                            <div className="flex justify-between text-[6px] items-baseline">
                                                                <span className="text-slate-500">{m.label}: <span className="text-slate-300 font-mono">{(item[m.key] || 0).toFixed(2)}</span></span>
                                                                <span className="font-mono text-slate-400">{((item[`${m.key}_similarity`] || 0) * 100).toFixed(0)}%</span>
                                                            </div>
                                                            <div className="h-0.5 w-full bg-slate-800/50 overflow-hidden">
                                                                <div className="h-full bg-slate-600" style={{ width: `${(item[`${m.key}_similarity`] || 0) * 100}%` }}></div>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>

                                            {/* SECTION 4: COLOR SPACES (ALL 60+ VARIANTS) */}
                                            <div className="mb-3">
                                                <div className="flex items-center gap-1 mb-1 border-b border-orange-500/20 pb-0.5">
                                                    <div className="w-2 h-2 rounded-full bg-gradient-to-tr from-red-500 via-green-500 to-blue-500"></div>
                                                    <span className="text-[7px] font-black text-orange-400 uppercase tracking-widest">Multi-Space Color Analysis</span>
                                                </div>
                                                <div className="flex flex-col gap-1.5">
                                                    {["rgb", "hsv", "lab", "ycrcb", "hls", "xyz", "gray"].map(space => (
                                                        <div key={space} className="bg-slate-900/30 p-1 border-l border-slate-800">
                                                            <div className="text-[6px] font-black text-slate-500 uppercase mb-1">{space} Domain</div>
                                                            <div className="grid grid-cols-3 gap-x-2 gap-y-1">
                                                                {['hist', 'cdf', 'joint'].map(type => (
                                                                    <div key={type} className="flex flex-col gap-0.5">
                                                                        <span className="text-[5px] text-slate-600 uppercase font-black">{type}</span>
                                                                        {['std', 'interp', 'gauss'].map(method => {
                                                                            const key = type === 'joint' ? `joint_${space}_${method}` : `${space}_${type}_${method}`;
                                                                            const sim = item[`${key}_similarity`];
                                                                            if (sim === undefined) return null;
                                                                            return (
                                                                                <div key={method} className="flex justify-between items-center text-[5px]">
                                                                                    <span className="text-slate-500 italic">{method}</span>
                                                                                    <span className="font-mono text-slate-400">{(sim * 100).toFixed(0)}%</span>
                                                                                </div>
                                                                            );
                                                                        })}
                                                                    </div>
                                                                ))}
                                                                <div className="flex flex-col gap-0.5">
                                                                    <span className="text-[5px] text-slate-600 uppercase font-black">Grid</span>
                                                                    <div className="flex justify-between items-center text-[5px]">
                                                                        <span className="text-slate-500 italic">cell_{space}</span>
                                                                        <span className="font-mono text-slate-400">{((item[`cell_${space}_similarity`] || 0) * 100).toFixed(0)}%</span>
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>

                                            {/* SECTION 5: HANDCRAFTED FEATURES (18) */}
                                            <div className="mb-1">
                                                <div className="flex items-center gap-1 mb-1 border-b border-emerald-500/20 pb-0.5">
                                                    <div className="w-2.5 h-2.5 border border-emerald-500/50 flex items-center justify-center text-[5px] font-black text-emerald-500">M</div>
                                                    <span className="text-[7px] font-black text-emerald-400 uppercase tracking-widest">Handcrafted Feature Mapping</span>
                                                </div>
                                                <div className="grid grid-cols-3 gap-1">
                                                    {[
                                                        'hog', 'hu_moments', 'lbp', 'color_moments', 'gabor', 'ccv', 
                                                        'zernike', 'geo', 'tamura', 'edge_orientation', 'glcm', 'wavelet', 
                                                        'correlogram', 'ehd', 'cld', 'spm', 'saliency'
                                                    ].map(feat => (
                                                        <div key={feat} className="bg-slate-900/50 p-1 border border-slate-800/50 flex flex-col items-center">
                                                            <span className="text-[5px] text-slate-600 font-black uppercase leading-none mb-1 text-center">{feat.replace('_', ' ')}</span>
                                                            <span className="text-[7px] font-mono text-emerald-500">{((item[`${feat}_similarity`] || 0) * 100).toFixed(0)}%</span>
                                                            <div className="w-full h-[1px] bg-slate-800 mt-0.5"><div className="h-full bg-emerald-700" style={{ width: `${(item[`${feat}_similarity`] || 0) * 100}%` }}></div></div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>
                                    </td>
                                    {enabledVis.histogram && (
                                        <td className="px-0 py-1 border-r border-slate-800/30">
                                            <div className="w-[600px] h-[180px] bg-slate-950 border-x border-slate-800 overflow-hidden">
                                                {renderHistogramGrid(item)}
                                            </div>
                                        </td>
                                    )}
                                    {enabledVis.hog && (
                                        <td className="px-0 py-1 border-r border-slate-800/30">
                                            <div className="w-[180px] h-[180px] bg-black border-x border-slate-800 overflow-hidden">
                                                <VisWrapper label="HOG Map" similarity={item.hog_similarity}>
                                                    {item.hogPreviewUrl ? <img src={item.hogPreviewUrl} className="w-full h-full object-cover" /> : <div className="w-full h-full flex items-center justify-center text-[9px] text-slate-800 uppercase">No Data</div>}
                                                </VisWrapper>
                                            </div>
                                        </td>
                                    )}
                                    {enabledVis.ccv && (
                                        <td className="px-0 py-1 border-r border-slate-800/30">
                                            <div className="w-[180px] h-[180px] bg-black border-x border-slate-800 overflow-hidden">
                                                <VisWrapper label="CCV Coherence" similarity={item.ccv_similarity}>
                                                    {item.ccvPreviewUrl ? <img src={item.ccvPreviewUrl} className="w-full h-full object-cover" /> : <div className="w-full h-full flex items-center justify-center text-[9px] text-slate-800 uppercase">No Data</div>}
                                                </VisWrapper>
                                            </div>
                                        </td>
                                    )}
                                    {enabledVis.gabor && (
                                        <td className="px-0 py-1 border-r border-slate-800/30">
                                            <div className="w-[180px] h-[180px] bg-black border-x border-slate-800 overflow-hidden">
                                                <VisWrapper label="Gabor Texture" similarity={item.gabor_similarity}>
                                                    {item.gaborPreviewUrl ? <img src={item.gaborPreviewUrl} className="w-full h-full object-cover" /> : <div className="w-full h-full flex items-center justify-center text-[9px] text-slate-800 uppercase">No Data</div>}
                                                </VisWrapper>
                                            </div>
                                        </td>
                                    )}
                                    {enabledVis.lbp && (
                                        <td className="px-0 py-1 border-r border-slate-800/30">
                                            <div className="w-[180px] h-[180px] bg-black border-x border-slate-800 overflow-hidden">
                                                <VisWrapper label="LBP Texture" similarity={item.lbp_similarity}>
                                                    {item.lbpPreviewUrl ? <img src={item.lbpPreviewUrl} className="w-full h-full object-cover" /> : <div className="w-full h-full flex items-center justify-center text-[9px] text-slate-800 uppercase">No Data</div>}
                                                </VisWrapper>
                                            </div>
                                        </td>
                                    )}
                                    {enabledVis.cellColor && (
                                        <td className="px-0 py-1 border-r border-slate-800/30">
                                            <div className="w-[180px] h-[180px] bg-black border-x border-slate-800 overflow-hidden">
                                                <VisWrapper label="Cell Grid" similarity={item.cell_rgb_similarity}>
                                                    {item.cellColorPreviewUrl ? <img src={item.cellColorPreviewUrl} className="w-full h-full object-cover" /> : <div className="w-full h-full flex items-center justify-center text-[9px] text-slate-800 uppercase">No Data</div>}
                                                </VisWrapper>
                                            </div>
                                        </td>
                                    )}
                                    {enabledVis.hu && (
                                        <td className="px-0 py-1">
                                            <div className="w-[180px] h-[180px] bg-black border-l border-slate-800 overflow-hidden">
                                                <VisWrapper label="Hu Moments" similarity={item.hu_moments_similarity}>
                                                    {item.huPreviewUrl ? <img src={item.huPreviewUrl} className="w-full h-full object-cover" /> : <div className="w-full h-full flex items-center justify-center text-[9px] text-slate-800 uppercase">No Data</div>}
                                                </VisWrapper>
                                            </div>
                                        </td>
                                    )}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            
        </Card>
    );
};

export default SimilarityResults;

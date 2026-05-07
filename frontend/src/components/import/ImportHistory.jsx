import React from 'react';
import useImageStore from '../../store/useImageStore';
import Card from '../common/Card';
import Badge from '../common/Badge';
import { formatDate } from '../../utils/formatters';
import { History, CheckCircle, XCircle, Clock, Image as ImageIcon, RefreshCw } from 'lucide-react';

const ImportHistory = () => {
    const { importHistory, recomputeAllFeatures, recomputing, pagination, loadImportHistory } = useImageStore();

    const statusIcons = {
        completed: <CheckCircle className="w-4 h-4" />,
        failed: <XCircle className="w-4 h-4" />,
        processing: <Clock className="w-4 h-4 animate-spin" />,
    };

    const statusVariants = {
        completed: 'success',
        failed: 'error',
        processing: 'warning',
    };

    return (
        <Card className="p-6">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <History className="w-5 h-5 text-primary-400" />
                    <h3 className="text-xl font-semibold text-white">Import Results & History</h3>
                </div>
                <button
                    onClick={recomputeAllFeatures}
                    disabled={recomputing || importHistory.length === 0}
                    className="flex items-center px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded border border-slate-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                >
                    <RefreshCw className={`w-4 h-4 mr-2 ${recomputing ? 'animate-spin' : ''}`} />
                    Recompute All
                </button>
            </div>

            {importHistory.length === 0 ? (
                <div className="text-center py-8 text-slate-400 bg-slate-800/20 rounded-lg border border-dashed border-slate-700">
                    <p>No import history yet. Drag and drop images above to start.</p>
                </div>
            ) : (
                <div className="flex flex-col gap-4 overflow-y-auto pr-2">
                    {importHistory.map((item) => (
                        <div
                            key={item.id}
                            className="bg-slate-800/50 border border-slate-700/50 rounded-lg overflow-hidden hover:border-slate-500/50 transition-all flex flex-row h-[380px] group"
                        >
                            {/* Image Preview - Left (LARGER) */}
                            <div className="relative w-[450px] shrink-0 bg-slate-900 flex items-center justify-center overflow-hidden border-r border-slate-700/50">
                                {item.previewUrl ? (
                                    <img
                                        src={item.previewUrl}
                                        alt={item.file_name}
                                        className="w-full h-full object-cover transition-transform group-hover:scale-105"
                                    />
                                ) : (
                                    <ImageIcon className="w-12 h-12 text-slate-600" />
                                )}

                                <div className="absolute top-2 right-2">
                                    <Badge variant={statusVariants[item.status || 'completed']}>
                                        {item.status || 'completed'}
                                    </Badge>
                                </div>
                            </div>

                            {/* Visual Features - Middle Column */}
                            <div className="w-[180px] shrink-0 p-4 border-r border-slate-700/50 flex flex-col gap-4 bg-slate-900/20">
                                <span className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1">Visual Features</span>
                                
                                {item.hogPreviewUrl && (
                                    <div className="flex flex-col gap-1">
                                        <span className="text-[10px] text-slate-400 font-bold">HOG Structure</span>
                                        <div className="w-full aspect-square rounded border border-slate-700 overflow-hidden bg-black shadow-inner">
                                            <img src={item.hogPreviewUrl} alt="HOG" className="w-full h-full object-contain" />
                                        </div>
                                    </div>
                                )}
                                
                                {item.huPreviewUrl && (
                                    <div className="flex flex-col gap-1">
                                        <span className="text-[10px] text-slate-400 font-bold">Hu Shape</span>
                                        <div className="w-full aspect-square rounded border border-slate-700 overflow-hidden bg-black shadow-inner">
                                            <img src={item.huPreviewUrl} alt="Hu" className="w-full h-full object-contain" />
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Metadata & Stats - Right Column */}
                            <div className="p-5 flex-1 flex flex-col min-w-0">
                                <div className="mb-6">
                                    <h4 className="font-bold text-white text-lg mb-1 truncate" title={item.file_name}>
                                        {item.file_name}
                                    </h4>
                                    <div className="flex items-center gap-3">
                                        <span className="text-xs text-slate-400 font-mono bg-slate-900 px-2 py-0.5 rounded">
                                            {item.width} × {item.height}
                                        </span>
                                        <span className="text-[10px] text-slate-500 font-medium">
                                            Added: {item.created_at ? formatDate(item.created_at) : 'Just now'}
                                        </span>
                                    </div>
                                </div>

                                <div className="flex-1 flex flex-col gap-6">
                                    <span className="text-[10px] text-slate-500 uppercase font-black tracking-widest block border-b border-slate-700/50 pb-1">Image Properties</span>
                                    
                                    <div className="grid grid-cols-2 gap-x-8 gap-y-4">
                                        {/* Brightness */}
                                        <div className="flex flex-col">
                                            <div className="flex justify-between items-center mb-1">
                                                <span className="text-slate-500 text-xs">Brightness</span>
                                                <span className="text-slate-200 text-xs font-mono">{(item.brightness * 100).toFixed(1)}%</span>
                                            </div>
                                            <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                                                <div className="h-full bg-primary-500" style={{ width: `${item.brightness * 100}%` }} />
                                            </div>
                                        </div>

                                        {/* Contrast */}
                                        <div className="flex flex-col">
                                            <div className="flex justify-between items-center mb-1">
                                                <span className="text-slate-500 text-xs">Contrast</span>
                                                <span className="text-slate-200 text-xs font-mono">{(item.contrast * 100).toFixed(1)}%</span>
                                            </div>
                                            <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                                                <div className="h-full bg-primary-500" style={{ width: `${item.contrast * 100}%` }} />
                                            </div>
                                        </div>

                                        {/* Saturation */}
                                        <div className="flex flex-col">
                                            <div className="flex justify-between items-center mb-1">
                                                <span className="text-slate-500 text-xs">Saturation</span>
                                                <span className="text-slate-200 text-xs font-mono">
                                                    {item.saturation !== undefined ? (item.saturation * 100).toFixed(1) + '%' : 'N/A'}
                                                </span>
                                            </div>
                                            <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                                                <div className="h-full bg-primary-500" style={{ width: `${(item.saturation || 0) * 100}%` }} />
                                            </div>
                                        </div>

                                        {/* Edge Density */}
                                        <div className="flex flex-col">
                                            <div className="flex justify-between items-center mb-1">
                                                <span className="text-slate-500 text-xs">Edge Density</span>
                                                <span className="text-slate-200 text-xs font-mono">{(item.edge_density * 100).toFixed(1)}%</span>
                                            </div>
                                            <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                                                <div className="h-full bg-primary-500" style={{ width: `${item.edge_density * 100}%` }} />
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Pagination Controls */}
            {pagination.pages > 1 && (
                <div className="flex items-center justify-between mt-8 pt-6 border-t border-slate-700/50">
                    <div className="text-sm text-slate-500">
                        Showing <span className="text-slate-300 font-bold">{importHistory.length}</span> of <span className="text-slate-300 font-bold">{pagination.total}</span> images
                    </div>
                    
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => loadImportHistory(pagination.page - 1, pagination.size)}
                            disabled={pagination.page <= 1}
                            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg border border-slate-700 disabled:opacity-30 disabled:cursor-not-allowed transition-all text-sm font-bold"
                        >
                            Previous
                        </button>
                        
                        <div className="flex items-center gap-1">
                            {[...Array(Math.min(5, pagination.pages))].map((_, i) => {
                                let pageNum = pagination.page;
                                if (pagination.page <= 3) pageNum = i + 1;
                                else if (pagination.page >= pagination.pages - 2) pageNum = pagination.pages - 4 + i;
                                else pageNum = pagination.page - 2 + i;
                                
                                if (pageNum <= 0 || pageNum > pagination.pages) return null;

                                return (
                                    <button
                                        key={pageNum}
                                        onClick={() => loadImportHistory(pageNum, pagination.size)}
                                        className={`w-10 h-10 flex items-center justify-center rounded-lg border transition-all text-sm font-bold ${
                                            pagination.page === pageNum
                                                ? 'bg-primary-600 border-primary-500 text-white shadow-lg shadow-primary-900/40'
                                                : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-500 hover:text-slate-200'
                                        }`}
                                    >
                                        {pageNum}
                                    </button>
                                );
                            })}
                        </div>

                        <button
                            onClick={() => loadImportHistory(pagination.page + 1, pagination.size)}
                            disabled={pagination.page >= pagination.pages}
                            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg border border-slate-700 disabled:opacity-30 disabled:cursor-not-allowed transition-all text-sm font-bold"
                        >
                            Next
                        </button>
                    </div>

                    <div className="text-sm text-slate-500">
                        Page <span className="text-slate-300 font-bold">{pagination.page}</span> of <span className="text-slate-300 font-bold">{pagination.pages}</span>
                    </div>
                </div>
            )}
        </Card>
    );
};

export default ImportHistory;

import React, { useEffect } from 'react';
import useImageStore from '../store/useImageStore';
import ImportForm from '../components/import/ImportForm';
import ImportHistory from '../components/import/ImportHistory';
import { Upload } from 'lucide-react';

const ImportPage = () => {
    const { loadImportHistory, pagination, syncVlmData, resetDatabase, recomputing } = useImageStore();

    // Load import history on mount
    useEffect(() => {
        loadImportHistory(pagination.page, pagination.size);
    }, [loadImportHistory]);

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between bg-slate-800/50 p-4 rounded-lg border border-slate-700">
                <h2 className="text-lg font-medium text-slate-300">
                    Total Images in Database: <span className="text-white font-bold text-2xl ml-2">{pagination.total}</span>
                </h2>
                <div className="flex items-center gap-4">
                    <button 
                        onClick={() => resetDatabase()}
                        disabled={recomputing}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all border ${
                            recomputing 
                                ? 'bg-slate-800 text-slate-500 border-slate-700 cursor-not-allowed' 
                                : 'bg-red-900/20 hover:bg-red-900/40 text-red-400 border-red-900/50 hover:border-red-800'
                        }`}
                    >
                        Reset Database
                    </button>
                    <button 
                        onClick={() => syncVlmData()}
                        disabled={recomputing}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                            recomputing 
                                ? 'bg-slate-700 text-slate-400 cursor-not-allowed' 
                                : 'bg-primary-600 hover:bg-primary-500 text-white shadow-lg shadow-primary-900/20'
                        }`}
                    >
                        {recomputing ? (
                            <span className="flex items-center gap-2">
                                <div className="w-4 h-4 border-2 border-slate-400 border-t-transparent rounded-full animate-spin"></div>
                                Syncing...
                            </span>
                        ) : (
                            'Sync VLM/Embeddings'
                        )}
                    </button>
                    <button 
                        onClick={() => loadImportHistory()}
                        className="text-primary-400 hover:text-primary-300 text-sm font-medium"
                    >
                        Refresh
                    </button>
                </div>
            </div>
            <ImportForm />
            <ImportHistory />
        </div>
    );
};

export default ImportPage;

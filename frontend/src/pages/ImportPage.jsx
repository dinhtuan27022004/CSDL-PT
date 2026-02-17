import React, { useEffect } from 'react';
import useImageStore from '../store/useImageStore';
import ImportForm from '../components/import/ImportForm';
import ImportHistory from '../components/import/ImportHistory';
import { Upload } from 'lucide-react';

const ImportPage = () => {
    const { loadImportHistory } = useImageStore();

    // Load import history on mount
    useEffect(() => {
        loadImportHistory();
    }, [loadImportHistory]);

    return (
        <div className="space-y-6">

            <ImportForm />
            <ImportHistory />

        </div>
    );
};

export default ImportPage;

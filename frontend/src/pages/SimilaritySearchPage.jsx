import React from 'react';
import SimilaritySearchForm from '../components/search/SimilaritySearchForm';
import SimilarityResults from '../components/search/SimilarityResults';

const SimilaritySearchPage = () => {
    return (
        <div className="space-y-6">
            <SimilaritySearchForm />
            <SimilarityResults />
        </div>
    );
};

export default SimilaritySearchPage;

def _retrieve_and_index_sources(self, sources):
    """
    Retrieve documents from sources and index them.
    
    Args:
        sources: List of source types to retrieve from
    """
    all_chunks = []
    
    # Use ThreadPoolExecutor for parallel document retrieval
    with ThreadPoolExecutor(max_workers=min(len(sources), 3)) as executor:
        future_to_source = {}
        
        # Start document retrieval for each source
        for source in sources:
            if source in self.document_sources:
                source_manager = self.document_sources[source]
                
                # Define a wrapper function that captures the current source value
                def get_docs_from_source(sm, src):
                    try:
                        logger.info(f"Getting documents from {src}")
                        return sm.get_documents()
                    except Exception as e:
                        logger.error(f"Error in get_documents for {src}: {str(e)}")
                        return []
                
                # Pass both source_manager and current source value
                future = executor.submit(get_docs_from_source, source_manager, source)
                future_to_source[future] = source
        
        # Process results as they complete
        for future in future_to_source:
            source = future_to_source[future]
            try:
                documents = future.result()
                if documents:
                    # Process and chunk documents
                    chunks = self.process_documents(documents, source_type=source)
                    all_chunks.extend(chunks)
                    logger.info(f"Retrieved and processed {len(documents)} documents from {source}")
            except Exception as e:
                logger.error(f"Error retrieving documents from {source}: {e}")
    
    # Index all chunks
    if all_chunks:
        self.index_documents(all_chunks)
        logger.info(f"Indexed {len(all_chunks)} chunks from {len(sources)} sources")

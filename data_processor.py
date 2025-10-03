import pandas as pd
import io
import re
from typing import List, Dict, Any, Optional

class DataProcessor:
    """Utility class for processing CSV data and text preprocessing"""
    
    def __init__(self):
        self.required_columns = ['title', 'content']
        self.optional_columns = ['id', 'url', 'source', 'published_date', 'created_at']
    
    def process_csv(self, csv_content: bytes) -> pd.DataFrame:
        """Process uploaded CSV content into a pandas DataFrame"""
        try:
            # Try to read CSV with different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(io.StringIO(csv_content.decode(encoding)))
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not decode CSV file with any supported encoding")
            
            # Basic validation
            if df.empty:
                raise ValueError("CSV file is empty")
            
            # Check for required columns
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                # Try to suggest column mappings
                suggestions = self._suggest_column_mappings(df.columns, missing_columns)
                error_msg = f"Missing required columns: {missing_columns}"
                if suggestions:
                    error_msg += f". Suggested mappings: {suggestions}"
                raise ValueError(error_msg)
            
            # Clean and preprocess the data
            df = self._clean_dataframe(df)
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error processing CSV: {str(e)}")
    
    def _suggest_column_mappings(self, available_columns: List[str], missing_columns: List[str]) -> Dict[str, str]:
        """Suggest column mappings based on available columns"""
        suggestions = {}
        
        column_mappings = {
            'title': ['headline', 'article_title', 'subject', 'name'],
            'content': ['text', 'body', 'description', 'article_content', 'summary'],
            'id': ['article_id', 'identifier', 'index'],
            'url': ['link', 'web_url', 'article_url'],
            'source': ['publication', 'publisher', 'site'],
            'published_date': ['date', 'timestamp', 'created', 'published'],
            'created_at': ['created', 'timestamp', 'date_added']
        }
        
        for missing_col in missing_columns:
            for available_col in available_columns:
                available_lower = available_col.lower()
                if available_lower in column_mappings.get(missing_col, []):
                    suggestions[missing_col] = available_col
                    break
        
        return suggestions
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the DataFrame"""
        df = df.copy()
        
        # Fill NaN values
        for col in ['title', 'content']:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
        
        # Remove rows with empty title or content
        df = df[df['title'].str.strip() != '']
        df = df[df['content'].str.strip() != '']
        
        # Apply text preprocessing
        if 'title' in df.columns:
            df['title_clean'] = df['title'].apply(self._preprocess_text)
        
        if 'content' in df.columns:
            df['content_clean'] = df['content'].apply(self._preprocess_text)
        
        # Create combined text for analysis
        df['combined_text'] = df['title_clean'] + ' ' + df['content_clean']
        
        # Add text statistics
        df['title_length'] = df['title_clean'].str.len()
        df['content_length'] = df['content_clean'].str.len()
        df['combined_length'] = df['combined_text'].str.len()
        df['word_count'] = df['combined_text'].str.split().str.len()
        
        # Remove very short articles (likely not useful)
        df = df[df['word_count'] >= 5]
        
        return df
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def validate_csv_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate CSV structure and return validation results"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check for required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check for empty dataframe
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("DataFrame is empty")
        
        # Check for empty title or content
        if 'title' in df.columns:
            empty_titles = df['title'].isna().sum() + (df['title'].str.strip() == '').sum()
            if empty_titles > 0:
                validation_results['warnings'].append(f"{empty_titles} articles have empty titles")
        
        if 'content' in df.columns:
            empty_content = df['content'].isna().sum() + (df['content'].str.strip() == '').sum()
            if empty_content > 0:
                validation_results['warnings'].append(f"{empty_content} articles have empty content")
        
        # Add statistics
        validation_results['statistics'] = {
            'total_rows': len(df),
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict()
        }
        
        return validation_results
    
    def get_sample_data(self, df: pd.DataFrame, n_samples: int = 3) -> List[Dict[str, Any]]:
        """Get sample data from the DataFrame"""
        sample_df = df.head(n_samples)
        
        samples = []
        for _, row in sample_df.iterrows():
            sample = {
                'title': str(row.get('title', '')),
                'content': str(row.get('content', ''))[:200] + '...' if len(str(row.get('content', ''))) > 200 else str(row.get('content', '')),
            }
            
            # Add other available columns
            for col in self.optional_columns:
                if col in row and pd.notna(row[col]):
                    sample[col] = str(row[col])
            
            samples.append(sample)
        
        return samples
    
    def export_results(self, df: pd.DataFrame, output_path: str) -> str:
        """Export processed DataFrame to CSV"""
        try:
            df.to_csv(output_path, index=False)
            return f"Results exported successfully to {output_path}"
        except Exception as e:
            raise ValueError(f"Error exporting results: {str(e)}")

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Union, Any
import warnings
warnings.filterwarnings('ignore')

class BankingDataCleaner:
    """
    A comprehensive cleaning tool for messy banking transaction data.
    Handles standardization of column names, date formats, currency values,
    and flags incomplete or corrupt rows.
    """
    
    def __init__(self):
        # Standard column mappings
        self.column_mapping = {
            'user id': 'user_id',
            'userid': 'user_id',
            'processed on': 'processed_date',
            'settlement date': 'settlement_date',
            'settled date': 'settlement_date_alt',
            'transaction date': 'transaction_date',
            'txn date': 'transaction_date_alt',
            'amount': 'amount',
            'currency': 'currency',
            'transaction type': 'transaction_type',
            'merchant name': 'merchant_name',
            'merchant category': 'merchant_category',
            'description': 'description',
            'location': 'location',
            'transaction id': 'transaction_id',
            'channel': 'channel',
            'account number': 'account_number'
        }
        
        # Critical columns that shouldn't be null
        self.critical_columns = ['user_id', 'amount', 'transaction_type', 'merchant_name']
        
    def clean_data(self, file_path: str) -> pd.DataFrame:
        """
        Main cleaning function that processes the entire dataset.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Cleaned pandas DataFrame
        """
        print("🧹 Starting data cleaning process...")
        
        # Load data
        df = self._load_data(file_path)
        print(f"📊 Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Clean column names
        df = self._standardize_columns(df)
        print("✅ Standardized column names")
        
        # Clean and normalize dates
        df = self._normalize_dates(df)
        print("✅ Normalized date formats")
        
        # Clean currency amounts
        df = self._normalize_amounts(df)
        print("✅ Converted currency values to floats")
        
        # Improve merchant names
        df = self._improve_merchant_names(df)
        print("✅ Improved merchant names from descriptions")
        
        # Flag problematic rows
        df = self._flag_incomplete_rows(df)
        print("✅ Flagged incomplete/corrupt rows")
        
        # Final cleanup
        df = self._final_cleanup(df)
        print("✅ Final cleanup completed")
        
        # Summary statistics
        self._print_summary(df)
        
        return df
    
    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Load Excel file and handle basic data issues."""
        try:
            df = pd.read_excel(file_path)
            return df
        except Exception as e:
            raise ValueError(f"Error loading file: {e}")
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to consistent format."""
        # Clean column names: lowercase, replace spaces with underscores
        df.columns = df.columns.str.lower().str.strip()
        
        # Apply mappings
        df.columns = [self.column_mapping.get(col, col.replace(' ', '_')) for col in df.columns]
        
        # Remove duplicate columns, keeping the first occurrence
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    
    def _normalize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all date columns to consistent YYYY-MM-DD format."""
        date_columns = [col for col in df.columns if 'date' in col]
        
        for col in date_columns:
            df[col] = df[col].apply(self._parse_date)
        
        # Consolidate redundant date columns
        df = self._consolidate_date_columns(df)
        
        return df
    
    def _parse_date(self, date_str: Any) -> Union[str, None]:
        """Parse various date formats into standardized YYYY-MM-DD format."""
        if pd.isna(date_str) or date_str is None:
            return None
            
        date_str = str(date_str).strip()
        
        # Common date patterns
        patterns = [
            # ISO format
            r'(\d{4}-\d{2}-\d{2})',
            # DD/MM/YYYY or MM/DD/YYYY
            r'(\d{1,2})/(\d{1,2})/(\d{4})',
            # DD-MM-YYYY or MM-DD-YYYY  
            r'(\d{1,2})-(\d{1,2})-(\d{4})',
            # Month DD, YYYY
            r'([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})',
            # DD-Mon-YYYY
            r'(\d{1,2})-([A-Za-z]+)-(\d{4})',
        ]
        
        try:
            # Try ISO format first
            if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                return pd.to_datetime(date_str[:10]).strftime('%Y-%m-%d')
            
            # Try parsing with pandas (handles many formats automatically)
            parsed = pd.to_datetime(date_str, errors='coerce', dayfirst=True)
            if not pd.isna(parsed):
                return parsed.strftime('%Y-%m-%d')
                
        except:
            pass
            
        return None  # Return None if parsing fails
    
    def _consolidate_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Consolidate redundant date columns into primary ones."""
        # Consolidate settlement dates
        if 'settlement_date' in df.columns and 'settlement_date_alt' in df.columns:
            df['settlement_date'] = df['settlement_date'].fillna(df['settlement_date_alt'])
            df = df.drop('settlement_date_alt', axis=1)
        
        # Consolidate transaction dates
        if 'transaction_date' in df.columns and 'transaction_date_alt' in df.columns:
            df['transaction_date'] = df['transaction_date'].fillna(df['transaction_date_alt'])
            df = df.drop('transaction_date_alt', axis=1)
        
        return df
    
    def _normalize_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert currency amounts to consistent float format and extract currency."""
        if 'amount' not in df.columns:
            return df
            
        df['amount_original'] = df['amount'].copy()  # Keep original for reference
        
        # Parse amounts and extract currencies
        parsed_data = df['amount'].apply(self._parse_amount_with_currency)
        df['amount'] = parsed_data.apply(lambda x: x[0] if x else None)
        df['amount_ccy'] = parsed_data.apply(lambda x: x[1] if x else None)
        
        # Fill missing amount_ccy with currency column if available
        if 'currency' in df.columns:
            df['amount_ccy'] = df['amount_ccy'].fillna(df['currency'])
        
        return df
    
    def _parse_amount_with_currency(self, amount_str: Any) -> Union[tuple, None]:
        """Parse currency formats and extract both amount and currency."""
        if pd.isna(amount_str) or amount_str is None:
            return None
            
        amount_str = str(amount_str).strip()
        original_amount = amount_str
        
        # Extract currency information
        extracted_currency = None
        
        # Check for currency symbols
        currency_symbols = {
            '€': 'EUR', '$': 'USD', '£': 'GBP', '¥': 'JPY', 
            '₹': 'INR', '₽': 'RUB', 'CHF': 'CHF'
        }
        
        for symbol, code in currency_symbols.items():
            if symbol in amount_str:
                extracted_currency = code
                break
        
        # Check for currency codes (3-letter codes at the end)
        currency_match = re.search(r'\b([A-Z]{3})\b', amount_str)
        if currency_match and not extracted_currency:
            potential_currency = currency_match.group(1)
            if potential_currency in ['EUR', 'USD', 'GBP', 'JPY', 'CHF']:
                extracted_currency = potential_currency
        
        # Clean amount for parsing
        clean_amount = re.sub(r'[€$£¥₹₽]', '', amount_str)
        clean_amount = re.sub(r'\s+[A-Z]{3}$', '', clean_amount)  # Remove trailing currency codes
        clean_amount = clean_amount.strip()
        
        # Handle different decimal separators
        # European format: 1.234,56 -> 1234.56
        if re.match(r'^-?\d{1,3}(\.\d{3})*,\d{2}$', clean_amount):
            clean_amount = clean_amount.replace('.', '').replace(',', '.')
        # Remove thousand separators (commas)
        elif ',' in clean_amount and '.' in clean_amount:
            # Format like 1,234.56
            clean_amount = clean_amount.replace(',', '')
        elif clean_amount.count(',') == 1 and '.' not in clean_amount:
            # European decimal comma: 123,45
            if len(clean_amount.split(',')[1]) <= 2:
                clean_amount = clean_amount.replace(',', '.')
            else:
                # Thousand separator: 1,234
                clean_amount = clean_amount.replace(',', '')
        
        try:
            parsed_amount = float(clean_amount)
            return (parsed_amount, extracted_currency)
        except:
            return None
    
    def _improve_merchant_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Improve merchant names by extracting from descriptions and fixing typos."""
        if 'merchant_name' not in df.columns:
            return df
            
        merchant_patterns = {
            'Amazon': ['amazon', 'ama on', 'ama#on', 'amaxon'],
            'Uber': ['uber', 'ube ', 'ube,', 'ube'],
            'Netflix': ['netflix', 'net@lix', 'net lix', 'netlix'],
            'H&M': ['h&m', 'h&mx', 'h&m@', 'h&m#'],
            'Lufthansa': ['lufthansa', 'luf@hansa', 'lufhansa'],
            'Apple': ['apple', 'app#e', 'appxe', 'appe'],
            'McDonald\'s': ['mcdonald', 'mcdonalds', 'mcd#nald', 'mcd nald', 'mcdnald'],
            'Shell': ['shell', 'shel,', 'shel ', 'shel'],
            'Starbucks': ['starbucks', 'stabucks', 'stabucks'],
            'Zara': ['zara', 'zar#', 'zar@', 'zarx']
        }
        
        # Track improvements
        improved_count = 0
        
        # First, handle missing merchant names
        missing_merchant_mask = df['merchant_name'].isna() | (df['merchant_name'] == '')
        
        for idx, row in df[missing_merchant_mask].iterrows():
            description = str(row['description']).lower() if pd.notna(row['description']) else ''
            
            for merchant_name, patterns in merchant_patterns.items():
                for pattern in patterns:
                    if pattern in description:
                        df.at[idx, 'merchant_name'] = merchant_name
                        improved_count += 1
                        break
                if df.at[idx, 'merchant_name'] == merchant_name:
                    break
        
        # Also improve existing merchant names that might have typos
        for idx, row in df.iterrows():
            current_merchant = str(row['merchant_name']).lower() if pd.notna(row['merchant_name']) else ''
            description = str(row['description']).lower() if pd.notna(row['description']) else ''
            
            # Skip if already a clean merchant name
            if current_merchant in [name.lower() for name in merchant_patterns.keys()]:
                continue
                
            # Check if current merchant name matches any pattern (for typo correction)
            for merchant_name, patterns in merchant_patterns.items():
                for pattern in patterns:
                    if pattern in current_merchant or pattern in description:
                        if df.at[idx, 'merchant_name'] != merchant_name:
                            df.at[idx, 'merchant_name'] = merchant_name
                            improved_count += 1
                        break
                if df.at[idx, 'merchant_name'] == merchant_name:
                    break
        
        if improved_count > 0:
            print(f"  • Improved {improved_count} merchant names from descriptions and typo corrections")
        
        return df
    
    def _flag_incomplete_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag rows with missing critical data or other issues."""
        df['data_quality_flags'] = ''
        flags = []
        
        for idx, row in df.iterrows():
            row_flags = []
            
            # Check for missing critical fields
            for col in self.critical_columns:
                if col in df.columns and (pd.isna(row[col]) or row[col] is None or row[col] == ''):
                    row_flags.append(f'missing_{col}')
            
            # Check for invalid amounts
            if 'amount' in df.columns and pd.isna(row['amount']):
                row_flags.append('invalid_amount')
            
            # Check for missing dates
            date_cols = [col for col in df.columns if 'date' in col and col != 'data_quality_flags']
            if all(pd.isna(row[col]) for col in date_cols):
                row_flags.append('no_valid_dates')
            
            df.at[idx, 'data_quality_flags'] = ';'.join(row_flags) if row_flags else 'clean'
        
        return df
    
    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup and column ordering."""
        # Standard column order
        preferred_order = [
            'user_id', 'transaction_id', 'transaction_date', 'settlement_date', 
            'processed_date', 'amount', 'amount_ccy', 'currency', 'transaction_type', 
            'merchant_name', 'merchant_category', 'description', 'location', 
            'channel', 'account_number', 'data_quality_flags'
        ]
        
        # Reorder columns
        existing_cols = [col for col in preferred_order if col in df.columns]
        other_cols = [col for col in df.columns if col not in existing_cols]
        df = df[existing_cols + other_cols]
        
        return df
    
    def _print_summary(self, df: pd.DataFrame) -> None:
        """Print cleaning summary statistics."""
        print("\n📋 CLEANING SUMMARY")
        print("=" * 50)
        print(f"Total rows processed: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        
        if 'data_quality_flags' in df.columns:
            clean_rows = (df['data_quality_flags'] == 'clean').sum()
            flagged_rows = len(df) - clean_rows
            print(f"Clean rows: {clean_rows} ({clean_rows/len(df)*100:.1f}%)")
            print(f"Flagged rows: {flagged_rows} ({flagged_rows/len(df)*100:.1f}%)")
            
            # Show flag distribution
            if flagged_rows > 0:
                print("\nData quality issues found:")
                flag_counts = {}
                for flags in df['data_quality_flags']:
                    if flags != 'clean':
                        for flag in flags.split(';'):
                            flag_counts[flag] = flag_counts.get(flag, 0) + 1
                
                for flag, count in sorted(flag_counts.items()):
                    print(f"  • {flag}: {count} rows")
        
        # Handle date range calculation safely
        if 'transaction_date' in df.columns:
            # Filter out null/invalid dates for range calculation
            valid_dates = df['transaction_date'].dropna()
            valid_dates = valid_dates[valid_dates != '']
            if len(valid_dates) > 0:
                try:
                    # Convert to datetime for proper comparison
                    date_series = pd.to_datetime(valid_dates, errors='coerce')
                    date_series = date_series.dropna()
                    if len(date_series) > 0:
                        min_date = date_series.min().strftime('%Y-%m-%d')
                        max_date = date_series.max().strftime('%Y-%m-%d')
                        print(f"\nDate range: {min_date} to {max_date}")
                    else:
                        print(f"\nDate range: No valid dates found")
                except:
                    print(f"\nDate range: Unable to calculate (mixed data types)")
            else:
                print(f"\nDate range: No dates available")
        
        # Handle amount range calculation safely
        if 'amount' in df.columns:
            valid_amounts = df['amount'].dropna()
            if len(valid_amounts) > 0:
                try:
                    min_amount = valid_amounts.min()
                    max_amount = valid_amounts.max()
                    print(f"Amount range: ${min_amount:.2f} to ${max_amount:.2f}")
                except:
                    print(f"Amount range: Unable to calculate")
    
    def save_cleaned_data(self, df: pd.DataFrame, output_path: str) -> None:
        """Save cleaned data to CSV."""
        df.to_csv(output_path, index=False)
        print(f"💾 Cleaned data saved to: {output_path}")


# Usage example
def main():
    """Example usage of the cleaning tool."""
    cleaner = BankingDataCleaner()
    
    # Clean the data
    cleaned_df = cleaner.clean_data('messy_banking_transactions.xlsx')
    
    # Save cleaned data
    cleaner.save_cleaned_data(cleaned_df, 'cleaned_banking_transactions.csv')
    
    # Optional: Show sample of cleaned data
    print("\n📊 SAMPLE OF CLEANED DATA")
    print("=" * 50)
    print(cleaned_df.head(10).to_string())
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = main()
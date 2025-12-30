from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer



class DataPreprocessor:
    """کلاس پیش‌پردازش داده‌ها"""
    
    def __init__(self, variance_threshold=0.01):
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.selector = VarianceThreshold(threshold=variance_threshold)
        self.selected_features = None
        
    def fit_transform(self, X, y=None):
        """آموزش و تبدیل داده‌ها"""
        
        # 1. مدیریت مقادیر گمشده
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        # 2. انتخاب ویژگی
        X_selected = self.selector.fit_transform(X_imputed)
        self.selected_features = self.selector.get_support(indices=True)
        
        # 3. مقیاس‌بندی
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # 4. کدگذاری برچسب (اگر y داده شده)
        if y is not None:
            y_encoded = self.label_encoder.fit_transform(y)
            return X_scaled, y_encoded
            
        return X_scaled
    
    def transform(self, X, y=None):
        """تبدیل داده‌های جدید با پارامترهای یادگرفته شده"""
        
        # 1. مدیریت مقادیر گمشده
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        # 2. انتخاب ویژگی‌های مشابه
        X_selected = X_imputed[:, self.selected_features]
        
        # 3. مقیاس‌بندی با پارامترهای ذخیره شده
        X_scaled = self.scaler.transform(X_selected)
        
        if y is not None:
            y_encoded = self.label_encoder.transform(y)
            return X_scaled, y_encoded
            
        return X_scaled
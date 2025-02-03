class I18nAuto:
    def __init__(self, language=None):
        """Placeholder i18n class that does not perform translation."""
        pass

    def __call__(self, key):
        """Returns the key itself as the translation."""
        return key

    def __repr__(self):
        return "I18nAuto Placeholder: No translation enabled"
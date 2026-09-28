from urielplus import urielplus

u = urielplus.URIELPlus()
u.reset()

# Configuration
u.set_cache(True)
u.set_aggregation('U')

# Integrate databases, excluding Glottolog to keep only the 8,172 source-integrated
# languages (Glottolog integration adds ~19,000 dialect rows used only
# as BFS targets for genetic imputation; see Section 2).
u.integrate_custom_databases("UPDATED_SAPHON", "BDPROTO", "GRAMBANK", "APICS", "EWAVE")

# Aggregates (union), runs BFS genetic imputation (fill_with_base_lang,
# default True), converts -1 to NaN, and fills remaining missing values
# with SoftImpute. Output is binary, matching URIEL+'s data convention.
u.softimpute_imputation()
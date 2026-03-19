import pomegranate as pg

# TEST 1: Does the 'distributions' module exist at all?
try:
    dist_module = pg.distributions
    print(f"pg.distributions exists: {dist_module}")
except AttributeError:
    print("pg.distributions DOES NOT exist (API changed).")
    dist_module = pg

# TEST 2: Check for the classes in the expected or new location
classes_to_check = ['StudentTDistribution', 'ProductDistribution']

for cls_name in classes_to_check:
    # Try the nested location (old way)
    try:
        cls = getattr(pg.distributions, cls_name)
        print(f"  {cls_name}: Found at pg.distributions.{cls_name} (OLD)")
    except (AttributeError, ModuleNotFoundError):
        # Try the top-level location (new way)
        try:
            cls = getattr(pg, cls_name)
            print(f"  {cls_name}: Found at pg.{cls_name} (NEW)")
        except AttributeError:
            print(f"  {cls_name}: NOT FOUND in either location.")

# Exit the interpreter
exit()
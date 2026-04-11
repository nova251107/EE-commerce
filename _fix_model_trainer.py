"""Fix model_trainer.py _try_load_persisted method."""
with open('model_trainer.py', 'r', encoding='utf-8') as f:
    src = f.read()

OLD = '''    def _try_load_persisted(self) -> bool:
        """Return True if cached models were loaded from disk."""
        try:
            pm_path = os.path.join(MODEL_DIR, "pricing_model.pkl")
            rm_path = os.path.join(MODEL_DIR, "recom_model.pkl")
            cw_path = os.path.join(MODEL_DIR, "category_weights.json")
            if not (os.path.exists(pm_path) and os.path.exists(rm_path)):
                return False
            with open(pm_path, "rb") as f:
                self.pricing = pickle.load(f)
            with open(rm_path, "rb") as f:
                self.recom = pickle.load(f)
            if os.path.exists(cw_path):
                with open(cw_path) as f:
                    self.optimiser._category_weights = json.load(f)
            self.optimiser.pricing = self.pricing
            self.optimiser.recom   = self.recom
            logger.info("ModelRegistry: loaded persisted models from disk")
            return True
        except Exception as e:
            logger.warning("ModelRegistry: could not load persisted models (%s)", e)
            return False'''

NEW = '''    def _try_load_persisted(self) -> bool:
        """Return True if cached models were loaded from disk."""
        try:
            pm_path = os.path.join(MODEL_DIR, "pricing_model.pkl")
            rm_path = os.path.join(MODEL_DIR, "recom_model.pkl")
            cw_path = os.path.join(MODEL_DIR, "category_weights.json")
            if not (os.path.exists(pm_path) and os.path.exists(rm_path)):
                return False

            with open(pm_path, "rb") as f:
                pm_data = pickle.load(f)
            if isinstance(pm_data, dict):
                self.pricing._model     = pm_data["model"]
                self.pricing._scaler    = pm_data["scaler"]
                self.pricing._alpha     = pm_data["alpha"]
                self.pricing._beta      = pm_data["beta"]
                self.pricing._gamma     = pm_data["gamma"]
                self.pricing._intercept = pm_data.get("intercept", 0.0)
                self.pricing._ready     = True
            else:
                logger.warning("ModelRegistry: pricing pkl format unrecognised")
                return False

            with open(rm_path, "rb") as f:
                rm_data = pickle.load(f)
            if isinstance(rm_data, dict):
                self.recom._model  = rm_data["model"]
                self.recom._scaler = rm_data["scaler"]
                self.recom._w1     = rm_data["w1"]
                self.recom._w2     = rm_data["w2"]
                self.recom._w3     = rm_data["w3"]
                self.recom._ready  = True
            else:
                logger.warning("ModelRegistry: recom pkl format unrecognised")
                return False

            if os.path.exists(cw_path):
                with open(cw_path) as f:
                    self.optimiser._category_weights = json.load(f)

            self.optimiser.pricing = self.pricing
            self.optimiser.recom   = self.recom
            logger.info("ModelRegistry: loaded persisted models from disk")
            return True
        except Exception as e:
            logger.warning("ModelRegistry: could not load persisted models (%s)", e)
            return False'''

if OLD in src:
    src2 = src.replace(OLD, NEW, 1)
    with open('model_trainer.py', 'w', encoding='utf-8') as f:
        f.write(src2)
    print("Patched OK")
else:
    # Try to find and print what's there
    idx = src.find('_try_load_persisted')
    print("Not found! Context at idx", idx)
    print(repr(src[max(0,idx-10):idx+500]))

import ast
try:
    ast.parse(open('model_trainer.py').read())
    print("Syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")

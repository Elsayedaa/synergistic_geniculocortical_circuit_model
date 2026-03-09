def get_single_unit_prediction_params(row, model, summary_data, prediction_params):
    try:
        fixef = {
            'intercept':summary_data[f'm{model}_fixed_effects'].loc[
                summary_data[f'm{model}_fixed_effects']['name'] == '(Intercept)'
            ]['fixef'].values[0],

            'slope': summary_data[f'm{model}_fixed_effects'].loc[
                summary_data[f'm{model}_fixed_effects']['name'] == 'sf'
            ]['fixef'].values[0],

            'intercept_v1_enh': summary_data[f'm{model}_fixed_effects'].loc[
                summary_data[f'm{model}_fixed_effects']['name'] == 'brain_areav1'
            ]['fixef'].values[0],

            'intercept_lgn_sup': summary_data[f'm{model}_fixed_effects'].loc[
                summary_data[f'm{model}_fixed_effects']['name'] == 'signsuppression'
            ]['fixef'].values[0],

            'slope_v1_enh': summary_data[f'm{model}_fixed_effects'].loc[
                summary_data[f'm{model}_fixed_effects']['name'] == 'sf:brain_areav1'
            ]['fixef'].values[0],

            'slope_lgn_sup': summary_data[f'm{model}_fixed_effects'].loc[
                summary_data[f'm{model}_fixed_effects']['name'] == 'sf:signsuppression'
            ]['fixef'].values[0],

            'intercept_v1_sup': summary_data[f'm{model}_fixed_effects'].loc[
                summary_data[f'm{model}_fixed_effects']['name'] == 'brain_areav1:signsuppression'
            ]['fixef'].values[0],

            'slope_v1_sup': summary_data[f'm{model}_fixed_effects'].loc[
                summary_data[f'm{model}_fixed_effects']['name'] == 'sf:brain_areav1:signsuppression'
            ]['fixef'].values[0]
        }

        ind = row.name
        brain_area = summary_data[f'data_{model}'].groupby(['unit']).agg(list).iloc[ind].brain_area[0]
        for sign in ['enhancement', 'suppression']:
            group = f'{brain_area}_{sign}'

            # get the random effects
            intercept = row['condval_(Intercept)']
            intercept_suppression = row['condval_signsuppression']
            slope = row['condval_sf']
            slope_suppression = row['condval_sf:signsuppression']

            # get the effect standard deviations
            intercept_e = row['condsd_(Intercept)']
            intercept_suppression_e = row['condsd_signsuppression']
            slope_e = row['condsd_sf']
            slope_suppression_e = row['condsd_sf:signsuppression']

            # set up the prediction map
            predmap = {
                'lgn_enhancement': {
                    'intercept_u': fixef['intercept'] + intercept,
                    'slope_u': fixef['slope'] + slope,
                    'intercept_e': intercept_e,
                    'slope_e': slope_e
                },

                'lgn_suppression': {
                    'intercept_u': fixef['intercept'] + fixef['intercept_lgn_sup'] + intercept + intercept_suppression,
                    'slope_u': fixef['slope'] + fixef['slope_lgn_sup'] + slope + slope_suppression,
                    'intercept_e': intercept_e + intercept_suppression_e,
                    'slope_e': slope_e + slope_suppression_e
                },

                'v1_enhancement': {
                    'intercept_u': fixef['intercept'] + fixef['intercept_v1_enh'] + intercept,
                    'slope_u': fixef['slope'] + fixef['slope_v1_enh'] + slope,
                    'intercept_e': intercept_e,
                    'slope_e': slope_e
                },

                'v1_suppression': {
                    'intercept_u': fixef['intercept'] + fixef['intercept_lgn_sup'] + fixef['intercept_v1_sup'] + intercept + intercept_suppression,
                    'slope_u': fixef['slope'] + fixef['slope_lgn_sup'] + fixef['slope_v1_sup'] + slope + slope_suppression,
                    'intercept_e': intercept_e + intercept_suppression_e,
                    'slope_e': slope_e + slope_suppression_e
                },
            }

            prediction_params[f'model_{model}']['brain_area'].append(brain_area)
            prediction_params[f'model_{model}']['sign'].append(sign)
            prediction_params[f'model_{model}']['intercept_u'].append(predmap[group]['intercept_u'])
            prediction_params[f'model_{model}']['intercept_e'].append(predmap[group]['intercept_e'])
            prediction_params[f'model_{model}']['slope_u'].append(predmap[group]['slope_u'])
            prediction_params[f'model_{model}']['slope_e'].append(predmap[group]['slope_e'])
            
    except IndexError:
        fixef = {
            'intercept':summary_data[f'm{model}_fixed_effects'].loc[
                summary_data[f'm{model}_fixed_effects']['name'] == '(Intercept)'
            ]['fixef'].values[0],

            'slope': summary_data[f'm{model}_fixed_effects'].loc[
                summary_data[f'm{model}_fixed_effects']['name'] == 'sf'
            ]['fixef'].values[0],

            'intercept_v1': summary_data[f'm{model}_fixed_effects'].loc[
                summary_data[f'm{model}_fixed_effects']['name'] == 'brain_areav1'
            ]['fixef'].values[0],

            'slope_v1': summary_data[f'm{model}_fixed_effects'].loc[
                summary_data[f'm{model}_fixed_effects']['name'] == 'sf:brain_areav1'
            ]['fixef'].values[0],

        }

        ind = row.name
        brain_area = summary_data[f'data_{model}'].groupby(['unit']).agg(list).iloc[ind].brain_area[0]
        
        model_signmap = {
            4: 'enhancement',
            5: 'suppression'
        }

        group = f'{brain_area}_{model_signmap[model]}'

        # get the random effects
        intercept = row['condval_(Intercept)']
        slope = row['condval_sf']

        # get the effect standard deviations
        intercept_e = row['condsd_(Intercept)']
        slope_e = row['condsd_sf']

        # set up the prediction map
        predmap = {
            f'lgn_{model_signmap[model]}': {
                'intercept_u': fixef['intercept'] + intercept,
                'slope_u': fixef['slope'] + slope,
                'intercept_e': intercept_e,
                'slope_e': slope_e
            },

            f'v1_{model_signmap[model]}': {
                'intercept_u': fixef['intercept'] + fixef['intercept_v1'] + intercept,
                'slope_u': fixef['slope'] + fixef['slope_v1'] + slope,
                'intercept_e': intercept_e,
                'slope_e': slope_e
            },

        }

        prediction_params[f'model_{model}']['brain_area'].append(brain_area)
        prediction_params[f'model_{model}']['sign'].append(model_signmap[model])
        prediction_params[f'model_{model}']['intercept_u'].append(predmap[group]['intercept_u'])
        prediction_params[f'model_{model}']['intercept_e'].append(predmap[group]['intercept_e'])
        prediction_params[f'model_{model}']['slope_u'].append(predmap[group]['slope_u'])
        prediction_params[f'model_{model}']['slope_e'].append(predmap[group]['slope_e'])
    
def get_intercepts_and_slopes(model):
    intercept = summary_data[f'm{model}_intercept_pred']['emmean']
    intercept_err = (
        summary_data[f'm{model}_intercept_pred']['emmean']
        -summary_data[f'm{model}_intercept_pred']['asymp.LCL']
    )
    
    slope = summary_data[f'm{model}_slope_pred']['sf.trend']
    slope_err = (
        summary_data[f'm{model}_slope_pred']['sf.trend']
        -summary_data[f'm{model}_slope_pred']['asymp.LCL']
    )
    
    
    return intercept, intercept_err, slope, slope_err    
    
def get_regression_predictions(model):
    try:
        lgn_enh_pred = summary_data[f'm{model}_estimates'].loc[
            (summary_data[f'm{model}_estimates']['brain_area'] == 'lgn')
            & (summary_data[f'm{model}_estimates']['sign'] == 'enhancement')
        ]['yvar'].values

        lgn_enh_err = lgn_enh_pred - summary_data[f'm{model}_estimates'].loc[
            (summary_data[f'm{model}_estimates']['brain_area'] == 'lgn')
            & (summary_data[f'm{model}_estimates']['sign'] == 'enhancement')
        ]['LCL'].values

        lgn_sup_pred = summary_data[f'm{model}_estimates'].loc[
            (summary_data[f'm{model}_estimates']['brain_area'] == 'lgn')
            & (summary_data[f'm{model}_estimates']['sign'] == 'suppression')
        ]['yvar'].values

        lgn_sup_err = lgn_sup_pred - summary_data[f'm{model}_estimates'].loc[
            (summary_data[f'm{model}_estimates']['brain_area'] == 'lgn')
            & (summary_data[f'm{model}_estimates']['sign'] == 'suppression')
        ]['LCL'].values

        v1_enh_pred = summary_data[f'm{model}_estimates'].loc[
            (summary_data[f'm{model}_estimates']['brain_area'] == 'v1')
            & (summary_data[f'm{model}_estimates']['sign'] == 'enhancement')
        ]['yvar'].values

        v1_enh_err = v1_enh_pred - summary_data[f'm{model}_estimates'].loc[
            (summary_data[f'm{model}_estimates']['brain_area'] == 'v1')
            & (summary_data[f'm{model}_estimates']['sign'] == 'enhancement')
        ]['LCL'].values

        v1_sup_pred = summary_data[f'm{model}_estimates'].loc[
            (summary_data[f'm{model}_estimates']['brain_area'] == 'v1')
            & (summary_data[f'm{model}_estimates']['sign'] == 'suppression')
        ]['yvar'].values

        v1_sup_err = v1_sup_pred - summary_data[f'm{model}_estimates'].loc[
            (summary_data[f'm{model}_estimates']['brain_area'] == 'v1')
            & (summary_data[f'm{model}_estimates']['sign'] == 'suppression')
        ]['LCL'].values

        return (
        lgn_enh_pred, lgn_enh_err,
        lgn_sup_pred, lgn_sup_err,
        v1_enh_pred, v1_enh_err,
        v1_sup_pred, v1_sup_err,
        )
    except KeyError:
        
        lgn_pred = summary_data[f'm{model}_estimates'].loc[
            (summary_data[f'm{model}_estimates']['brain_area'] == 'lgn')
        ]['yvar'].values

        lgn_err = lgn_pred - summary_data[f'm{model}_estimates'].loc[
            (summary_data[f'm{model}_estimates']['brain_area'] == 'lgn')
        ]['LCL'].values

        v1_pred = summary_data[f'm{model}_estimates'].loc[
            (summary_data[f'm{model}_estimates']['brain_area'] == 'v1')
        ]['yvar'].values

        v1_err = v1_pred - summary_data[f'm{model}_estimates'].loc[
            (summary_data[f'm{model}_estimates']['brain_area'] == 'v1')
        ]['LCL'].values
        
        return (
        lgn_pred, lgn_err,
        v1_pred, v1_err,
        )
package com.rakeshmalik.insectid.adapters;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.rakeshmalik.insectid.R;
import com.rakeshmalik.insectid.pojo.InsectModel;
import java.util.List;

public class ModelSpinnerAdapter extends ArrayAdapter<InsectModel> {

    public ModelSpinnerAdapter(@NonNull Context context, @NonNull List<InsectModel> models) {
        super(context, 0, models);
    }

    @NonNull
    @Override
    public View getView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {
        return createView(position, convertView, parent);
    }

    @Override
    public View getDropDownView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {
        return createView(position, convertView, parent);
    }

    private View createView(int position, View convertView, ViewGroup parent) {
        if (convertView == null) {
            convertView = LayoutInflater.from(getContext()).inflate(R.layout.item_model_spinner, parent, false);
        }

        InsectModel model = getItem(position);
        if (model != null) {
            TextView modelName = convertView.findViewById(R.id.modelName);
            ImageView iconLegacy = convertView.findViewById(R.id.iconLegacy);
            ImageView iconExperimental = convertView.findViewById(R.id.iconExperimental);

            modelName.setText(model.getDisplayName());
            iconLegacy.setVisibility(model.isLegacy() ? View.VISIBLE : View.GONE);
            iconExperimental.setVisibility(model.isExperimental() ? View.VISIBLE : View.GONE);
        }

        return convertView;
    }
}
